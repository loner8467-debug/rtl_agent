"""Batch RTL Agent for VerilogEval spec-to-rtl.

Each problem must contain *_prompt.txt, *_ref.sv, and *_test.sv.
The reference source is used only by the verifier and is never sent to Qwen.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


MODEL_NAME = "Qwen/Qwen3.8-27B"
API_BASE_URL = "https://api-inference.modelscope.cn/v1"
DATASET_DIR = Path(r"D:\aidaima\ceshi\tiku\verilog-eval-main\dataset_spec-to-rtl")
OUTPUT_DIR = Path(r"D:\aidaima\ceshi\agent_outputs")


@dataclass(frozen=True)
class Problem:
    name: str
    prompt: Path
    reference: Path
    testbench: Path


def read_text(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"找不到文件：{path}")
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"文件为空：{path}")
    return text


def find_problems(dataset_dir: Path) -> list[Problem]:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到题库目录：{dataset_dir}")
    problems = []
    for prompt in sorted(dataset_dir.glob("*_prompt.txt")):
        prefix = prompt.name.removesuffix("_prompt.txt")
        reference = dataset_dir / f"{prefix}_ref.sv"
        testbench = dataset_dir / f"{prefix}_test.sv"
        if reference.is_file() and testbench.is_file():
            problems.append(Problem(prefix, prompt, reference, testbench))
        else:
            print(f"跳过不完整题目：{prefix}", file=sys.stderr)
    if not problems:
        raise ValueError("没有找到完整的 *_prompt.txt / *_ref.sv / *_test.sv 题目")
    return problems


def extract_rtl(text: str) -> str:
    match = re.search(
        r"```(?:systemverilog|verilog|sv)?\s*(.*?)```",
        text, flags=re.IGNORECASE | re.DOTALL,
    )
    code = (match.group(1) if match else text).strip()
    if not re.search(r"\bmodule\b", code):
        raise ValueError("模型输出中没有找到 module")
    return code + "\n"


def ask_model(client: OpenAI, question: str, old_code: str | None,
              feedback: str | None, timeout: float) -> str:
    if old_code is None:
        prompt = (
            "请解决下面的 RTL 题目。严格按照题目要求实现名为 TopModule 的模块，"
            "保持端口名、方向和位宽不变。只输出完整、可综合的 SystemVerilog 代码块，"
            "附带不超过三行的说明。\n\n题目：\n" + question
        )
    else:
        prompt = (
            "上一版 RTL 没有通过仿真，请根据反馈修复。严格保持题目规定的 TopModule "
            "接口不变，只输出修复后的完整、可综合 SystemVerilog 代码块和极短说明。\n\n"
            f"题目：\n{question}\n\n上一版代码：\n```systemverilog\n{old_code}\n```\n\n"
            f"验证反馈：\n{feedback}"
        )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": (
                "你是 RTL SystemVerilog 工程师。只生成可综合代码，注意复位、时序、"
                "位宽、边界条件和题目接口。"
            )},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=8192,
        timeout=max(1.0, timeout),
    )
    return response.choices[0].message.content or ""


def run(command: list[str], cwd: Path, timeout: float) -> tuple[int, str]:
    try:
        p = subprocess.run(
            command, cwd=cwd, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=max(1.0, timeout), check=False,
        )
        return p.returncode, p.stdout[-16000:]
    except FileNotFoundError as exc:
        return 127, f"找不到验证工具：{exc}"
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, f"验证工具超时\n{output[-16000:]}"


def verify(workdir: Path, generated: Path, problem: Problem,
           timeout: float) -> tuple[bool, str]:
    simv = workdir / "simv"
    rc, log = run(
        ["iverilog", "-g2012", "-s", "tb", "-o", str(simv),
         str(generated), str(problem.reference), str(problem.testbench)],
        workdir, timeout,
    )
    if rc != 0:
        return False, "编译失败：\n" + log
    rc, sim = run(["vvp", str(simv)], workdir, timeout)
    combined = (log + "\n" + sim).strip()
    if rc != 0:
        return False, "仿真进程失败：\n" + combined[-16000:]
    mismatch = re.search(r"Mismatches:\s*(\d+)", combined, re.IGNORECASE)
    if mismatch and int(mismatch.group(1)) == 0:
        return True, "验证通过：\n" + combined[-16000:]
    if re.search(r"TEST\s+PASSED", combined, re.IGNORECASE):
        return True, "验证通过：\n" + combined[-16000:]
    if mismatch:
        return False, "功能验证失败：\n" + combined[-16000:]
    return False, "未检测到明确的验证通过标志：\n" + combined[-16000:]


def solve(client: OpenAI, problem: Problem, output_dir: Path,
          max_attempts: int, time_limit: float) -> dict:
    question = read_text(problem.prompt)
    final_path = output_dir / f"{problem.name}.sv"
    run_dir = output_dir / ".runs" / problem.name / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + time_limit
    old_code = None
    feedback = None
    history = []

    for attempt in range(1, max_attempts + 1):
        left = deadline - time.monotonic()
        if left <= 0:
            break
        print(f"[{problem.name}] 尝试 {attempt}/{max_attempts}，剩余 {left:.1f} 秒")
        attempt_dir = run_dir / f"attempt_{attempt}"
        attempt_dir.mkdir()
        try:
            raw = ask_model(client, question, old_code, feedback, min(left, 120.0))
            code = extract_rtl(raw)
        except Exception as exc:
            feedback = f"模型调用或代码提取失败：{exc}"
            history.append({"attempt": attempt, "status": "model_error", "feedback": feedback})
            break
        generated = attempt_dir / "solution.sv"
        generated.write_text(code, encoding="utf-8")
        (attempt_dir / "model_output.txt").write_text(raw, encoding="utf-8")
        ok, feedback = verify(attempt_dir, generated, problem, min(max(1.0, deadline - time.monotonic()), 120.0))
        (attempt_dir / "feedback.txt").write_text(feedback, encoding="utf-8")
        history.append({"attempt": attempt, "status": "pass" if ok else "fail", "feedback": feedback})
        if ok:
            output_dir.mkdir(parents=True, exist_ok=True)
            final_path.write_text(code, encoding="utf-8")
            return {"problem": problem.name, "status": "solved", "attempts": attempt,
                    "output": str(final_path), "history": history}
        old_code = code

    return {"problem": problem.name, "status": "unsolved", "attempts": len(history),
            "output": None, "history": history}


def main() -> int:
    parser = argparse.ArgumentParser(description="VerilogEval RTL Agent")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--only", help="只运行一道题，例如 Prob001_zero")
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=300.0)
    args = parser.parse_args()
    if not os.environ.get("MODELSCOPE_API_KEY"):
        print("错误：请先设置 MODELSCOPE_API_KEY", file=sys.stderr)
        return 2
    try:
        problems = find_problems(args.dataset)
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2
    if args.only:
        problems = [p for p in problems if p.name == args.only]
        if not problems:
            print(f"错误：找不到题目 {args.only}", file=sys.stderr)
            return 2
    client = OpenAI(base_url=API_BASE_URL, api_key=os.environ["MODELSCOPE_API_KEY"])
    args.output.mkdir(parents=True, exist_ok=True)
    results = [solve(client, p, args.output, args.max_attempts, args.time_limit) for p in problems]
    report = args.output / "summary.json"
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    solved = sum(r["status"] == "solved" for r in results)
    print(f"完成：{solved}/{len(results)} 道题通过")
    print(f"汇总文件：{report}")
    return 0 if solved == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
