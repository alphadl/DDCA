# Release Notes

- **DCA/DDCA 核心**：`advantage_dca_grpo` / `advantage_dca_rloo`，条件长度分数（Z-score + Sigmoid），耦合奖励基线。**DDCA**（默认 use_dynamic=True）：长度优势按通过率 ρ=n/G 缩放（Difficulty-Aware Coefficient）。
- **指标**：AES、pass@K、avg_tokens。
- **数据与评估**：GSM8K/MATH 加载、答案等价性、`scripts/evaluate.py`。
- **校验**：`scripts/verify_dca.py`（参数失效、零和性质）、`tests/` 单元测试。
- **一键复现**：`BUILTIN_ONLY=1 TRAIN_SIZE=10 VAL_SIZE=5 ./scripts/run_full_pipeline.sh`。
- **VERL 接入**：见 [docs/INTEGRATION_VERL.md](docs/INTEGRATION_VERL.md)。依赖：Python ≥3.7，numpy，pyyaml。
