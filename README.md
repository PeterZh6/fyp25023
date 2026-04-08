# Budgeted Analysis Strategy Selection via Reinforcement Learning

Undergraduate final-year project (FYP): use reinforcement learning under a limited analysis budget to choose analysis depth for indirect control-flow sites in binaries, aiming to maximize the resolution success rate.

**Repository:** [github.com/PeterZh6/fyp25023](https://github.com/PeterZh6/fyp25023)

## Problem setting

Different reverse-engineering tools (Ghidra, angr, Pin, etc.) differ in time cost and success rate when resolving indirect jumps/calls. The agent must, under a **fixed budget**, pick an **analysis level** (or skip) for each site to maximize the number of successfully resolved sites. Environment parameters (difficulty distribution, success rates, costs) can be calibrated from real data and written to YAML, or left at defaults for quick experiments.

## Project layout

```
fyp/
├── rl/                              # Gymnasium env, baselines, training, plots
│   ├── budget_env.py                # EnvConfig, AnalysisBudgetEnv (reads YAML)
│   ├── baselines.py                 # Hand-crafted policies and rollouts
│   ├── train.py                     # PPO training, eval, experiment-suite CLI
│   ├── plotting.py                  # Training curves etc. (learning dynamics)
│   ├── plot_results.py              # Sensitivity / cross-binary → results/figures/
│   ├── run_sensitivity.py           # Train PPO + baseline eval under cost presets
│   ├── run_cross_binary.py          # Cross-binary generalization (load saved model)
│   ├── analyze_sanity.py            # Post-training sanity analysis
│   └── configs/env_configs.yaml     # Per-binary MDP params (from calibration pipeline)
│
├── calibration/                     # Derive env config from labeled data
│   ├── config.py                    # Shared filters (e.g. exclude specrand)
│   ├── compute_distributions.py     # Difficulty distributions
│   ├── compute_success_rates.py     # (level × difficulty) success rates
│   ├── configure_costs.py           # Cost ratios (manual or from logs)
│   ├── benchmark_tools.py           # Measure Ghidra/angr runtime (needs local tools/)
│   └── generate_env_config.py       # Merge into rl/configs/env_configs.yaml
│
├── extraction/                      # Extract indirect flows from binaries
│   ├── ghidra_scripts/
│   │   └── ExportIndirectFlowsCustom.java
│   ├── run_ghidra_batch.sh
│   ├── extract_angr_indirect.py     # angr CFGFast → JSON aligned with Ghidra
│   └── check_env.py
│
├── evaluation/
│   ├── compare_tools.py             # Ghidra vs angr comparison
│   └── generate_silver_labels.py    # Agreement / conflict → silver difficulty labels
│
├── data/
│   ├── binaries/                    # ARM32 ELF (gitignored by default)
│   ├── ghidra_results/              # *_ghidra.json (gitignored by default)
│   ├── angr_results/                # *_angr.json (gitignored by default)
│   ├── silver_labels/               # generate_silver_labels output (optional)
│   └── calibration/                 # Distribution / success JSON, sensitivity summaries, model zips, etc.
│
├── tests/                           # pytest (env, baselines, calibration logic)
├── results/                         # Training outputs; paper figures in results/figures/
├── scripts/run_sanity.sh            # Legacy batch example (args may not match current train CLI)
├── budget_env_rl.py                 # Early single-file script (reference only)
├── requirements.txt
└── README.md
```

## Environment and dependencies

```bash
pip install -r requirements.txt
```

Main dependencies: Gymnasium, NumPy, Matplotlib, stable-baselines3, angr. Ghidra / Pin are optional local tools; paths are listed at the end.

## Quick start: data → calibration → RL

### 1. Extract indirect flows

```bash
python extraction/check_env.py

bash extraction/run_ghidra_batch.sh data/binaries/<suite> data/ghidra_results

python extraction/extract_angr_indirect.py -d data/binaries/<suite> -o data/angr_results

python -m evaluation.compare_tools --all \
  --ghidra-dir data/ghidra_results --angr-dir data/angr_results
```

### 2. Silver labels and YAML environment (optional)

After you have both Ghidra and angr JSON, generate per-binary difficulty labels, run the calibration scripts, and produce `env_configs.yaml` (paths below are each command’s defaults and can be chained as-is):

```bash
python -m evaluation.generate_silver_labels \
  --ghidra-dir data/ghidra_results \
  --angr-dir data/angr_results \
  --output-dir data/silver_labels

python -m calibration.compute_distributions
python -m calibration.compute_success_rates
python -m calibration.configure_costs --manual
python -m calibration.generate_env_config
```

`configure_costs` also supports `--from-logs --ghidra-log ... --angr-log ...` to estimate L2 from timing logs; see each module’s `--help` for more flags. The repo already includes `rl/configs/env_configs.yaml` for reproducible training.

Optional: with Ghidra installed locally, run `python -m calibration.benchmark_tools` to produce `data/calibration/cost_benchmark.json` for reported runtimes.

### 3. RL training and experiment suite

Default config path: `--config rl/configs/env_configs.yaml`. Modes:

| Mode | Description |
|------|-------------|
| `sanity` | Short training + comparison to baselines |
| `train` | Multi-seed training, writes `eval_results.json` |
| `eval` | Load `--eval-model` and evaluate |
| `suite_a` | Multiple binaries × multiple seeds |
| `suite_b` | λ (cost penalty) sweep |
| `suite_c` | Observation ablation (site features / global stats / oracle) |
| `suite_d` | Cross-binary train split and zero-shot test |

Examples:

```bash
python -m rl.train --mode sanity --binary gcc

python -m rl.train --mode train --binary gcc --timesteps 500000 --num-seeds 5

python -m rl.train --mode suite_b --save-dir results/exp_b/

python -m rl.train --mode eval --binary ssh --eval-model path/to/model.zip
```

Common flags: `--budget-ratio`, `--cost-lambda`, `--no-site-features`, `--no-global-stats`, `--oracle`.

### 4. Cost sensitivity and cross-binary evaluation

```bash
python -m rl.run_sensitivity --help          # Summary → data/calibration/sensitivity_results.json

python -m rl.run_cross_binary --help         # Summary → data/calibration/cross_binary_results.json

python rl/plot_results.py                    # Read the JSON above → results/figures/*.png/pdf
```

### 5. Unit tests

```bash
pytest tests/
```

## Analysis levels (MDP concepts)

| Level | Example meaning | Default relative cost (overridable in YAML) |
|-------|-----------------|---------------------------------------------|
| L1    | Lightweight static (e.g. Ghidra) | 1 |
| L2    | Heavier static/symbolic (e.g. angr) | 5 |
| L3    | High cost (e.g. dynamic Pin) | 20 |
| SKIP  | Do not analyze | 0 |

Per difficulty bucket (easy / medium / hard), success probabilities per level are estimated by `compute_success_rates` (and related steps) and written to `env_configs.yaml`; training uses the YAML as ground truth.

## Main experiments

- **Multi-seed**: training variance (`train` / `suite_a`)
- **λ sweep**: cost–success tradeoff (`suite_b`)
- **Observation ablation**: site features, global stats, oracle upper bound (`suite_c`)
- **Cross-binary**: different train/test splits (`suite_d` and `run_cross_binary`)
- **Cost-preset sensitivity**: L2/L3 ratios such as `baseline` / `low` / `high` in `run_sensitivity`

## Tool paths (when installed locally)

- **Ghidra 12.0**: `tools/ghidra_12.0/` (scripts such as `benchmark_tools` use this path)
- **Intel Pin**: `tools/pin/`
- **angr**: via pip; indirect-flow extraction in `extraction/extract_angr_indirect.py`
