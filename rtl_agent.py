"""A small RTL generation-and-repair agent.

The agent reads one problem statement, asks a model for SystemVerilog, runs a
local simulator, and sends only the simulator feedback back for repair.  The
reference answer is passed to the verifier when requested; it is never put in
the model prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "Qwen/Qwen3.8-27B"
API_BASE_URL = "https://api-inference.modelscope.cn/v1"


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"文件为空：{path}")
    return text


def extract_rtl(text: str) -> str:
    """Extract the first fenced Verilog/SystemVerilog block, or plain code."""
    match = re.search(
        r"```(?:systemverilog|verilog|sv)?\s*(.*?)```", text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    code = match.group(1) if match else text
    code = code.strip()
    if not re.search(r"\bmodule\b", code):
        raise ValueError("模型输出中没有找到 module，无法作为 RTL 文件验证")
    return code + "\n"


def remaining_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def ask_model(client: OpenAI, question: str, previous_code: str | None,
              feedback: str | None, timeout_seconds: float) -> str:
    if previous_code is None:
        user_content = (
            "请解决下面的 RTL 题目。\n\n"
            f"题目：\n{question}\n\n"
            "要求：输出完整、可综合的 SystemVerilog。严格保持题目中的模块名、"
            "端口名、端口方向和位宽。只输出代码块和极短说明。"
        )
    else:
        user_content = (
            "下面的 RTL 方案验证失败，请根据验证反馈修复它。\n\n"
            f"原题目：\n{question}\n\n"
            f"上一版代码：\n```systemverilog\n{previous_code}\n```\n\n"
            f"验证反馈：\n{feedback}\n\n"
            "请输出修复后的完整 SystemVerilog。不要只给修改片段，严格保持接口不变，"
            "只输出代码块和极短说明。"
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是专业的 RTL SystemVerilog 工程师。代码必须可综合，"
                    "注意同步时序、复位、位宽、边界条件和接口协议。"
                ),
            },
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
        max_tokens=8192,
        timeout=max(1.0, timeout_seconds),
    )
    return response.choices[0].message.content or ""


def run_process(command: list[str], cwd: Path, timeout_seconds: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=max(1.0, timeout_seconds),
            check=False,
        )
        return completed.returncode, completed.stdout[-12000:]
    except FileNotFoundError as exc:
        return 127, f"找不到验证工具：{exc}"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, f"验证工具超时\n{output[-12000:]}"


def verify_with_iverilog(workdir: Path, generated: Path, testbench: Path,
                         reference: Path | None, timeout_seconds: float) -> tuple[bool, str]:
    """Generic Icarus verifier. Adjust source order/top module if the benchmark requires it."""
    sim_binary = workdir / "simv"
    sources = [str(generated), *( [str(reference)] if reference else [] ), str(testbench)]
    compile_command = ["iverilog", "-g2012", "-o", str(sim_binary), *sources]
    rc, log = run_process(compile_command, workdir, timeout_seconds)
    if rc != 0:
        return False, "编译失败：\n" + log

    rc, sim_log = run_process(["vvp", str(sim_binary)], workdir, timeout_seconds)
    combined = (log + "\n" + sim_log).strip()
    if rc != 0:
        return False, "仿真失败：\n" + combined[-12000:]
    return True, "验证通过：\n" + combined[-12000:]


def verify_with_vivado(workdir: Path, generated: Path, testbench: Path,
                       reference: Path | None, tb_top: str,
                       timeout_seconds: float) -> tuple[bool, str]:
    """Basic Vivado xsim verifier for a single testbench top module."""
    sources = [str(generated), *( [str(reference)] if reference else [] ), str(testbench)]
    rc, log = run_process(["xvlog", "--sv", *sources], workdir, timeout_seconds)
    if rc != 0:
        return False, "xvlog 编译失败：\n" + log

    snapshot = "rtl_agent_snapshot"
    rc, elab_log = run_process(
        ["xelab", tb_top, "-s", snapshot, "-debug", "typical"],
        workdir,
        timeout_seconds,
    )
    if rc != 0:
        return False, "xelab elaboration 失败：\n" + (log + "\n" + elab_log)[-12000:]

    rc, sim_log = run_process(
        ["xsim", snapshot, "-runall"],
        workdir,
        timeout_seconds,
    )
    combined = (log + "\n" + elab_log + "\n" + sim_log).strip()
    if rc != 0:
        return False, "xsim 仿真失败：\n" + combined[-12000:]
    return True, "验证通过：\n" + combined[-12000:]


def verify(workdir: Path, generated: Path, testbench: Path,
           reference: Path | None, simulator: str, tb_top: str,
           timeout_seconds: float) -> tuple[bool, str]:
    if simulator == "iverilog":
        return verify_with_iverilog(workdir, generated, testbench, reference, timeout_seconds)
    return verify_with_vivado(workdir, generated, testbench, reference, tb_top, timeout_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description="RTL 生成、验证和自动修复 Agent")
    parser.add_argument("--question", type=Path, required=True, help="题目文件")
    parser.add_argument("--testbench", type=Path, required=True, help="仿真文件")
    parser.add_argument("--reference", type=Path, help="参考答案；只交给验证器，不交给模型")
    parser.add_argument("--output", type=Path, required=True, help="最终 RTL 输出文件")
    parser.add_argument("--simulator", choices=["iverilog", "vivado"], default="iverilog")
    parser.add_argument("--tb-top", default="tb", help="Vivado 仿真顶层模块名")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=300.0, help="单题总时间，单位秒")
    args = parser.parse_args()

    api_key = os.environ.get("MODELSCOPE_API_KEY")
    if not api_key:
        print("错误：没有设置 MODELSCOPE_API_KEY", file=sys.stderr)
        return 2

    question = read_text(args.question)
    testbench = args.testbench.resolve()
    reference = args.reference.resolve() if args.reference else None
    if not testbench.is_file():
        print(f"找不到仿真文件：{testbench}", file=sys.stderr)
        return 2
    if reference and not reference.is_file():
        print(f"找不到参考答案：{reference}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_dir = args.output.parent / ".rtl_agent_runs" / args.question.stem
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    deadline = time.monotonic() + args.time_limit
    previous_code = None
    feedback = None
    history = []

    for attempt in range(1, args.max_attempts + 1):
        left = remaining_seconds(deadline)
        if left <= 0:
            break

        print(f"[{attempt}/{args.max_attempts}] 正在请求模型，剩余 {left:.1f} 秒...")
        try:
            raw = ask_model(client, question, previous_code, feedback, min(left, 120.0))
            code = extract_rtl(raw)
        except Exception as exc:
            feedback = f"模型调用或代码提取失败：{exc}"
            history.append({"attempt": attempt, "status": "model_error", "feedback": feedback})
            break

        attempt_dir = run_dir / f"attempt_{attempt}"
        attempt_dir.mkdir()
        generated = attempt_dir / "solution.sv"
        generated.write_text(code, encoding="utf-8")
        shutil.copy2(testbench, attempt_dir / testbench.name)
        if reference:
            shutil.copy2(reference, attempt_dir / reference.name)

        print(f"[{attempt}/{args.max_attempts}] 正在验证 RTL...")
        ok, feedback = verify(
            attempt_dir,
            generated,
            attempt_dir / testbench.name,
            (attempt_dir / reference.name) if reference else None,
            args.simulator,
            args.tb_top,
            min(remaining_seconds(deadline), 120.0),
        )
        history.append({"attempt": attempt, "status": "pass" if ok else "fail", "feedback": feedback})
        (attempt_dir / "model_output.txt").write_text(raw, encoding="utf-8")
        (attempt_dir / "feedback.txt").write_text(feedback, encoding="utf-8")

        if ok:
            args.output.write_text(code, encoding="utf-8")
            result = {"status": "solved", "attempts": attempt, "output": str(args.output), "history": history}
            (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"验证通过，最终结果已保存到：{args.output}")
            return 0

        previous_code = code
        if remaining_seconds(deadline) <= 0:
            break

    result = {"status": "unsolved", "attempts": len(history), "history": history}
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("在规定时间内没有得到验证通过的 RTL，结果为：无法解决")
    print(f"详细过程保存在：{run_dir}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
