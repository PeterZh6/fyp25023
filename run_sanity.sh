#!/bin/bash
set -e

echo "=== Multi-seed ==="
for seed in 0 1 2 3 4; do
    python budget_env_rl.py --mode train --algo ppo --seed $seed --out out_seed_$seed
done

echo "=== Lambda sweep ==="
for lam in 0.0 0.01 0.02 0.05 0.1; do
    python budget_env_rl.py --mode train --algo ppo --cost_lambda $lam --out out_lambda_$lam
done

echo "=== Info ablation ==="
python budget_env_rl.py --mode train --algo ppo --no_cluster_stats --out out_no_stats

echo "=== Oracle ==="
python budget_env_rl.py --mode train --algo ppo --oracle --out out_oracle

echo "=== Analyze ==="
python analyze_sanity.py

echo "DONE!"