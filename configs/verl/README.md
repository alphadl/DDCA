# VERL baseline configs

- **vanilla**: GRPO with 0/1 reward only (no length penalty).
- **grpo_lp**: GRPO with coupled length penalty; reward = (1 - γ·length) if correct else 0.
- **dca**: DCA advantage; reward = 0/1, advantage = DCA-GRPO (β=0.2).

Use these as reference overrides. VERL itself does not read YAML by default; you pass overrides on the command line. The run script `scripts/run_verl_baselines.sh` shows how to run each baseline with the same base command.
