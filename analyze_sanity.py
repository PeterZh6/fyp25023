"""
Sanity check analysis script.
Run after completing all experiments.

Usage:
    python analyze_sanity.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from budget_env_rl import (
    EnvConfig, AnalysisBudgetEnv, 
    AlwaysL1, eval_policy, load_rl, SB3PolicyAdapter
)
import dataclasses

# ============================================================
# 1. Multi-seed stability
# ============================================================
def analyze_multi_seed(seeds=[0,1,2,3,4], budget=60, n_eval=300):
    print("="*60)
    print("1. MULTI-SEED STABILITY")
    print("="*60)
    
    results = []
    for seed in seeds:
        model_path = f"out_seed_{seed}/best_model.zip"
        if not os.path.exists(model_path):
            model_path = f"out_seed_{seed}/model.zip"
        
        if not os.path.exists(model_path):
            print(f"  [SKIP] seed {seed}: model not found")
            continue
        
        model = load_rl("ppo", model_path)
        policy = SB3PolicyAdapter(model)
        
        cfg = EnvConfig(budget=budget, seed=seed)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results.append(m)
        print(f"  seed={seed}: solved={m['solved_mean']:.2f}, spent={m['spent_mean']:.1f}, return={m['return_mean']:.3f}")
    
    if len(results) > 1:
        solved_means = [r['solved_mean'] for r in results]
        return_means = [r['return_mean'] for r in results]
        print(f"\n  SUMMARY ({len(results)} seeds):")
        print(f"    Solved: {np.mean(solved_means):.2f} ± {np.std(solved_means):.2f}")
        print(f"    Return: {np.mean(return_means):.3f} ± {np.std(return_means):.3f}")
    
    return results

# ============================================================
# 2. Lambda sweep (Pareto curve)
# ============================================================
def analyze_lambda_sweep(lambdas=[0.0, 0.01, 0.02, 0.05, 0.1], budget=60, n_eval=300):
    print("\n" + "="*60)
    print("2. LAMBDA SWEEP (cost-utility tradeoff)")
    print("="*60)
    
    results = {}
    for lam in lambdas:
        model_path = f"out_lambda_{lam}/best_model.zip"
        if not os.path.exists(model_path):
            model_path = f"out_lambda_{lam}/model.zip"
        
        if not os.path.exists(model_path):
            print(f"  [SKIP] lambda={lam}: model not found")
            continue
        
        model = load_rl("ppo", model_path)
        policy = SB3PolicyAdapter(model)
        
        cfg = EnvConfig(budget=budget, cost_lambda=lam)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results[lam] = m
        print(f"  λ={lam}: solved={m['solved_mean']:.2f}, spent={m['spent_mean']:.1f}")
    
    # Plot Pareto curve
    if len(results) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        lams = sorted(results.keys())
        solved = [results[l]['solved_mean'] for l in lams]
        spent = [results[l]['spent_mean'] for l in lams]
        
        ax.scatter(spent, solved, s=100, zorder=5)
        for i, l in enumerate(lams):
            ax.annotate(f'λ={l}', (spent[i], solved[i]), 
                       textcoords="offset points", xytext=(5,5))
        
        ax.set_xlabel('Budget Spent (mean)')
        ax.set_ylabel('Targets Solved (mean)')
        ax.set_title('Lambda Sweep: Spend vs Solve Tradeoff')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('sanity_lambda_pareto.png', dpi=150)
        plt.close()
        print(f"\n  Saved: sanity_lambda_pareto.png")
    
    return results

# ============================================================
# 3. Info ablation
# ============================================================
def analyze_info_ablation(budget=60, n_eval=300):
    print("\n" + "="*60)
    print("3. INFO ABLATION (cluster stats)")
    print("="*60)
    
    results = {}
    
    # With cluster stats (baseline)
    baseline_path = "out_seed_0/best_model.zip"
    if not os.path.exists(baseline_path):
        baseline_path = "out_seed_0/model.zip"
    
    if os.path.exists(baseline_path):
        model = load_rl("ppo", baseline_path)
        policy = SB3PolicyAdapter(model)
        cfg = EnvConfig(budget=budget, use_cluster_stats=True)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results['with_stats'] = m
        print(f"  WITH cluster stats: solved={m['solved_mean']:.2f}, return={m['return_mean']:.3f}")
    
    # Without cluster stats (ablation)
    ablation_path = "out_no_stats/best_model.zip"
    if not os.path.exists(ablation_path):
        ablation_path = "out_no_stats/model.zip"
    
    if os.path.exists(ablation_path):
        model = load_rl("ppo", ablation_path)
        policy = SB3PolicyAdapter(model)
        cfg = EnvConfig(budget=budget, use_cluster_stats=False)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results['no_stats'] = m
        print(f"  WITHOUT cluster stats: solved={m['solved_mean']:.2f}, return={m['return_mean']:.3f}")
    
    # Comparison
    if 'with_stats' in results and 'no_stats' in results:
        diff_solved = results['with_stats']['solved_mean'] - results['no_stats']['solved_mean']
        diff_return = results['with_stats']['return_mean'] - results['no_stats']['return_mean']
        print(f"\n  DIFFERENCE (with - without):")
        print(f"    Solved: {diff_solved:+.2f}")
        print(f"    Return: {diff_return:+.3f}")
        
        if diff_solved > 0.5 or diff_return > 0.1:
            print("    → Information gain IS useful")
        else:
            print("    → Information gain has LIMITED effect")
    
    return results

# ============================================================
# 4. Oracle upper bound
# ============================================================
def analyze_oracle(budget=60, n_eval=300):
    print("\n" + "="*60)
    print("4. ORACLE UPPER BOUND")
    print("="*60)
    
    results = {}
    
    # Normal RL (baseline)
    baseline_path = "out_seed_0/best_model.zip"
    if not os.path.exists(baseline_path):
        baseline_path = "out_seed_0/model.zip"
    
    if os.path.exists(baseline_path):
        model = load_rl("ppo", baseline_path)
        policy = SB3PolicyAdapter(model)
        cfg = EnvConfig(budget=budget, oracle_mode=False)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results['normal'] = m
        print(f"  Normal RL: solved={m['solved_mean']:.2f}, return={m['return_mean']:.3f}")
    
    # Oracle RL
    oracle_path = "out_oracle/best_model.zip"
    if not os.path.exists(oracle_path):
        oracle_path = "out_oracle/model.zip"
    
    if os.path.exists(oracle_path):
        model = load_rl("ppo", oracle_path)
        policy = SB3PolicyAdapter(model)
        cfg = EnvConfig(budget=budget, oracle_mode=True)
        m = eval_policy(cfg, policy, n_episodes=n_eval, seed0=1000)
        results['oracle'] = m
        print(f"  Oracle RL: solved={m['solved_mean']:.2f}, return={m['return_mean']:.3f}")
    
    # Fixed-L1 baseline
    cfg = EnvConfig(budget=budget)
    m = eval_policy(cfg, AlwaysL1(), n_episodes=n_eval, seed0=1000)
    results['fixed_l1'] = m
    print(f"  Fixed-L1:  solved={m['solved_mean']:.2f}, return={m['return_mean']:.3f}")
    
    # Compute headroom
    if 'normal' in results and 'oracle' in results:
        headroom = results['oracle']['solved_mean'] - results['normal']['solved_mean']
        print(f"\n  HEADROOM (oracle - normal): {headroom:+.2f} targets")
        
        if headroom > 1.0:
            print("    → Significant room for improvement with better features")
        else:
            print("    → Normal RL is already close to oracle")
    
    return results

# ============================================================
# 5. Summary comparison bar chart
# ============================================================
def plot_summary(multi_seed, info_ablation, oracle_results):
    print("\n" + "="*60)
    print("5. GENERATING SUMMARY PLOT")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = []
    solved_means = []
    solved_stds = []
    
    # Fixed-L1
    if 'fixed_l1' in oracle_results:
        labels.append('Fixed-L1')
        solved_means.append(oracle_results['fixed_l1']['solved_mean'])
        solved_stds.append(oracle_results['fixed_l1']['solved_std'])
    
    # RL (multi-seed mean)
    if multi_seed:
        labels.append('RL-PPO\n(5 seeds)')
        means = [r['solved_mean'] for r in multi_seed]
        solved_means.append(np.mean(means))
        solved_stds.append(np.std(means))
    
    # No cluster stats
    if 'no_stats' in info_ablation:
        labels.append('RL-PPO\n(no stats)')
        solved_means.append(info_ablation['no_stats']['solved_mean'])
        solved_stds.append(info_ablation['no_stats']['solved_std'])
    
    # Oracle
    if 'oracle' in oracle_results:
        labels.append('RL-PPO\n(oracle)')
        solved_means.append(oracle_results['oracle']['solved_mean'])
        solved_stds.append(oracle_results['oracle']['solved_std'])
    
    x = np.arange(len(labels))
    bars = ax.bar(x, solved_means, yerr=solved_stds, capsize=5, 
                  color=['gray', 'steelblue', 'coral', 'green'][:len(labels)])
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Solved Targets (mean ± std)')
    ax.set_title('Sanity Check Summary (Budget=60)')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, solved_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sanity_summary.png', dpi=150)
    plt.close()
    print(f"  Saved: sanity_summary.png")

# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    multi_seed = analyze_multi_seed()
    lambda_results = analyze_lambda_sweep()
    info_ablation = analyze_info_ablation()
    oracle_results = analyze_oracle()
    
    plot_summary(multi_seed, info_ablation, oracle_results)
    
    print("\n" + "="*60)
    print("DONE. Check generated plots:")
    print("  - sanity_lambda_pareto.png")
    print("  - sanity_summary.png")
    print("="*60)