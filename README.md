# Budgeted Analysis Strategy Selection via Reinforcement Learning

Final Year Project (FYP): Using RL to intelligently allocate analysis budget across indirect control-flow targets in binary programs.

**Repository:** [github.com/PeterZh6/fyp25023](https://github.com/PeterZh6/fyp25023)

## Problem

Binary reverse-engineering tools (Ghidra, angr, Pin) vary in cost and accuracy when resolving indirect jumps/calls. Given a finite analysis budget, the agent must decide **which analysis level** to apply to each target to maximise the number of successfully resolved sites.

## Project Structure

```
fyp/
├── rl/                          # RL environment, baselines, training
│   ├── budget_env.py            # Gymnasium MDP environment (EnvConfig, AnalysisBudgetEnv)
│   ├── baselines.py             # Hand-crafted policies + rollout utilities
│   ├── train.py                 # PPO/DQN training, model loading, CLI
│   ├── plotting.py              # Learning curve, solved-vs-budget, policy behavior
│   └── analyze_sanity.py        # Post-training sanity check analysis
│
├── extraction/                  # Binary analysis data extraction
│   ├── ghidra_scripts/
│   │   └── ExportIndirectFlowsCustom.java  # Ghidra headless script (indirect jumps + calls → JSON)
│   ├── run_ghidra_batch.sh      # Batch driver for Ghidra headless analysis
│   ├── extract_angr.py          # Indirect flow extraction via angr CFGFast
│   └── check_env.py             # Environment check utility
│
├── evaluation/                  # Result evaluation and tool comparison
│   └── compare_tools.py         # Ghidra vs angr indirect flow comparison
│
├── data/                        # Dataset (binaries gitignored, results regenerable)
│   ├── binaries/                # ARM32 ELF binaries (gitignored)
│   │   ├── coreutils_gcc_O2/
│   │   ├── coreutils_gcc_O3/
│   │   ├── cpu2006_gcc_O3/
│   │   ├── openssl_gcc_O3/
│   │   └── ssh_servers_gcc_O3/
│   ├── ghidra_results/          # ExportIndirectFlows output (gitignored)
│   └── angr_results/            # extract_angr.py output (gitignored)
│
├── scripts/                     # Shell automation
│   └── run_sanity.sh            # Batch training + analysis driver
│
├── archive/                     # Old/archived scripts
│
├── budget_env_rl.py             # Original monolithic script (kept for reference)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
```

### 1. Extract indirect flows from binaries

```bash
# Check angr environment
python extraction/check_env.py

# Run Ghidra headless on a binary suite
bash extraction/run_ghidra_batch.sh data/binaries/ssh_servers_gcc_O3 data/ghidra_results

# Run angr extraction
python extraction/extract_angr.py -d data/binaries/ssh_servers_gcc_O3 -o data/angr_results

# Compare Ghidra vs angr results
python -m evaluation.compare_tools ssh
```

### 2. RL training

```bash
# Sanity check: evaluate baseline policies
python -m rl.train --mode sanity

# Train PPO agent
python -m rl.train --mode train --algo ppo --timesteps 200000

# Sweep across budgets
python -m rl.train --mode sweep --algo ppo --model out/model.zip

# Full experiment suite (multi-seed, lambda sweep, ablations)
bash scripts/run_sanity.sh
```

## Analysis Levels

| Level | Tool Example     | Cost | Success Rate (easy/medium/hard) |
|-------|------------------|------|---------------------------------|
| L1    | Ghidra (static)  | 1    | 80% / 40% / 10%                |
| L2    | angr (symbolic)  | 5    | 95% / 80% / 50%                |
| L3    | Pin (dynamic)    | 20   | 100% / 95% / 85%               |
| SKIP  | —                | 0    | —                               |

## Key Experiments

- **Multi-seed stability**: 5 random seeds to measure training variance
- **Lambda sweep**: cost penalty λ ∈ {0, 0.01, 0.02, 0.05, 0.1} for Pareto front
- **Info ablation**: with/without cluster-level statistics in observation
- **Oracle upper bound**: reveal hidden difficulty to estimate headroom

## Tools

- **Ghidra 12.0** (`tools/ghidra_12.0/`): static binary analysis
- **Intel Pin** (`tools/pin/`): dynamic binary instrumentation
- **angr**: symbolic execution-based CFG recovery
