# RTL 生成、验证和修复 Agent

这个示例把一题的流程串起来：

```text
题目文件 -> Qwen -> solution.sv -> 仿真/编译 -> 错误反馈 -> Qwen 修复
```

参考答案只传给验证器，不放进模型提示词。

## 运行前

```powershell
pip install -U openai
$env:MODELSCOPE_API_KEY = "你的新Token"
```

开发阶段可以先用 Icarus Verilog 验证：

```powershell
python .\rtl_agent.py `
  --question "D:\aidaima\ceshi\questions\sync_fifo.txt" `
  --testbench "D:\aidaima\ceshi\questions\sync_fifo_tb.sv" `
  --reference "D:\aidaima\ceshi\questions\sync_fifo_answer.sv" `
  --output "D:\aidaima\ceshi\outputs\sync_fifo.sv" `
  --simulator iverilog `
  --max-attempts 5 `
  --time-limit 300
```

如果题库的 testbench 已经通过 `include` 或其他方式引用参考答案，运行时可以省略
`--reference`，避免重复编译。

有 Vivado 后，可以把验证器切换为 xsim：

```powershell
python .\rtl_agent.py `
  --question "D:\aidaima\ceshi\questions\sync_fifo.txt" `
  --testbench "D:\aidaima\ceshi\questions\sync_fifo_tb.sv" `
  --reference "D:\aidaima\ceshi\questions\sync_fifo_answer.sv" `
  --output "D:\aidaima\ceshi\outputs\sync_fifo.sv" `
  --simulator vivado `
  --tb-top tb
```

`--tb-top` 必须改成 testbench 中真正的顶层模块名。

## 说明

这个版本是开发原型。正式比赛时，ModelScope 远程 API 不能作为断网沙箱中的最终推理服务；
需要把 `OpenAI` 客户端的地址换成本地模型服务，并保留同样的 Agent、验证和日志结构。
