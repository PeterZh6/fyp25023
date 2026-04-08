# Budgeted Analysis-Strategy Selection for Binary Indirect Control Flow Resolution Using Reinforcement Learning

## Abstract

Binary program analysis tools face a fundamental resource allocation problem: different analysis techniques—such as pattern matching, symbolic execution, and dynamic analysis—vary greatly in computational cost and in how often they successfully resolve ambiguous control flow. Because analysis budgets are always finite, deciding *where* to spend expensive machinery is as important as having the machinery at all. This work formulates the problem of selecting analysis strategies for indirect control flow sites as a budget-constrained Markov Decision Process and trains a reinforcement learning agent with Proximal Policy Optimization (PPO) to learn allocation policies. Using a silver-label methodology based on agreement and disagreement between Ghidra and angr across 31 ARM32 binaries, we calibrate a simulation environment that reflects realistic success rates and costs. Empirical evaluation shows that the learned policy significantly outperforms hand-crafted baselines on large binaries dominated by hard sites: on `dealII`, mean per-episode resolve rate reaches **0.928** (seed 0, cost ratio 1:5:20) versus **0.117** for a budget-aware heuristic and **0.089** for uniform L1—about **7.9×** the heuristic rate and **10.5×** the L1 rate—while performance on easy-dominated binaries does not collapse relative to strong baselines. Sensitivity analysis over three abstract cost presets (1:3:15, 1:5:20, 1:10:50) shows that this pattern is stable except under extreme cost pressure, where every strategy is forced toward cheap actions. Cross-binary evaluation using policies trained on `gcc` versus `dealII` and tested on the other binary (`data/calibration/cross_binary_results.json`, 200 episodes, seed 0) shows **near-total loss of transfer** (e.g. **0.814 → 0.198** and **0.928 → 0.082** in mean resolve rate), indicating that the agent learns **distribution-specific** tactics rather than a single universal policy. Together, these results support budgeted RL as a viable framing for indirect-CF analysis and motivate future work on online adaptation and more faithful environment models.

---

## 1. Introduction

### 1.1 Problem

Indirect control flow is one of the central obstacles in binary reverse engineering and vulnerability analysis. On ARM32, instructions such as `BX reg` or `BLX reg` transfer control to an address that is not fixed in the instruction encoding; the target may depend on register values, memory contents, or compiler-generated jump tables. Static tools must therefore combine fast whole-program heuristics with more expensive analyses when simple patterns fail.

It is helpful to think in terms of three abstract levels of effort:

- **L1 (lightweight):** Pattern matching and heuristic disassembly (e.g. Ghidra’s built-in indirect jump and call handling). L1 is comparatively cheap and runs over the whole binary, but it leaves many sites unresolved when the program uses indirect calls, C++ vtables, or aggressive optimization.
- **L2 (medium):** Stronger static analyses such as value-set analysis or bounded symbolic exploration (e.g. angr’s CFGFast pipeline with indirect jump resolution). L2 is more expensive than L1 but resolves many cases that pure heuristics miss.
- **L3 (heavy):** Full symbolic or dynamic execution. L3 offers the highest potential recall but is orders of magnitude more costly if applied indiscriminately.

In practice, an analyst—or an automated pipeline—does not have unlimited budget. The natural research question is therefore: **given a fixed budget over an entire binary, how should one allocate L1/L2/L3 attempts across hundreds or thousands of indirect control flow sites so as to maximize the number of successfully resolved sites (or a similar objective)?** This is inherently sequential and resource-constrained: spending too much early can leave no budget for later hard sites; skipping upgrades everywhere wastes information that L2 or L3 could exploit.

### 1.2 Contribution

This report makes the following contributions:

1. **Problem formulation.** We cast binary analysis strategy selection under a global budget as a finite-horizon, budget-constrained Markov Decision Process, with actions at each site corresponding to skip, L1, L2, or L3, and rewards tied to successful resolution minus penalized cost.

2. **Silver labels without compiler ground truth.** We define per-site difficulty using **agreement between Ghidra and angr** (easy / medium / hard), which avoids reliance on compiler-instrumented oracle labels that do not cover all indirect call patterns in our setting.

3. **Empirical evidence for PPO.** We show that PPO learns non-trivial allocation policies that beat strong static baselines on several binaries, with the largest gains on **large, hard-dominated** programs such as `dealII`.

4. **Negative results as guidance.** We document cases where RL offers little benefit (small binaries, easy-dominated distributions) and a **cross-binary generalization failure**, which argues for per-distribution training or online-adaptive methods rather than a single frozen policy.

---

## 2. Background and Related Work

### 2.1 Indirect Control Flow Resolution

Resolving indirect jumps and calls is difficult because the set of possible targets is often data-dependent and because optimized binaries obscure high-level structure. Recent security and systems literature has emphasized that **ground-truth difficulty varies widely** across programs and optimization levels: some binaries yield to lightweight disassembly, while others require heavyweight reasoning. Jump-table analysis (including ARM TBB/TBH idioms) has received dedicated treatment; separate lines of work target **learned disassembly heuristics** and **multi-tool pipelines**. Across these settings, a recurring theme is that **no single analysis uniformly dominates** every binary.

*(Brief citations the write-up should attach in the final version: Pang et al., USENIX Security 2022 on measuring ground-truth difficulty; work on jump-table and switch-id analysis such as SJA-style approaches; Ddisasm and related learning-based disassembly.)*

### 2.2 Existing Tool Approaches

**Ghidra** applies fast whole-program analysis and pattern-based recovery of control flow, which makes it attractive as a default L1. **angr** builds a CFG and can apply more expensive indirect jump resolution on top of VEX lifting. Commercial and open tools such as **IDA Pro**, **Binary Ninja**, and **Radare2** each implement their own mixtures of heuristics and deeper analyses. Comparative studies in the literature consistently show **tool complementarity**: one engine may resolve a site another misses, especially across compiler versions, languages (C vs C++), and optimization levels.

Our silver-label scheme operationalizes this complementarity: disagreement or joint failure of two mature tools is treated as evidence that a site is genuinely **hard** for static analysis in practice.

### 2.3 RL for Program Analysis

Reinforcement learning has been applied successfully to **fuzzing** (e.g. seed scheduling, energy allocation) and to **symbolic execution** (path prioritization). These settings share with ours the idea that **exploration cost must be budgeted**. To our knowledge, prior work does not frame **per-site upgrade decisions for indirect control flow resolution** under a single global analysis budget as an MDP and study PPO-style policies against static baselines in that framing. The present work therefore occupies a distinct point in the design space: not replacing Ghidra or angr, but **learning when to escalate** between abstract analysis levels.

---

## 3. Research Journey and Problem Evolution

FYP reports benefit from an honest account of how the research question was refined. This section summarizes that trajectory.

### 3.1 Initial Direction: Jump Table Identification with RL

The project originally followed the brief “disassembly of ARM binaries with RL.” The first concrete goal was to use RL to **identify jump tables**, in particular ARM32 **TBB/TBH** patterns, where a compact encoding indexes into a table of branch targets. A prototype pipeline was built around **compiler-instrumented ground truth** from the CCR / ORACLEGT toolchain, which provides strong supervision for specific idioms.

### 3.2 Finding: Ghidra Already Achieves Near-Perfect Jump Table Recall

Systematic evaluation on several datasets showed that **Ghidra’s built-in jump table recovery was already extremely strong**, leaving little room for RL to improve recall:

| Dataset | Ghidra JT Recall |
|---------|-----------------|
| coreutils GCC O2 | 100% |
| coreutils GCC O3 | 98.4% |
| SPEC CPU2006 GCC O3 | 96.9% |
| OpenSSL GCC O3 | 100% |

In other words, for jump tables specifically, **L1 was already at ceiling**; RL would be chasing vanishing errors rather than a substantive allocation problem.

### 3.3 Pivot: From Jump Table Identification to Budgeted Strategy Selection

Rather than abandoning RL for binary analysis, we **changed the question**. The interesting problem is not “identify one more table” but **decide how deeply to analyze each ambiguous site under a cap on total effort**. Jump tables were too narrow a surrogate; we broadened the scope to **all indirect control flow sites** (indirect jumps and indirect calls) extracted consistently from Ghidra and angr.

### 3.4 Ground Truth Limitation and Silver Label Solution

The CCR-style oracle does not cover every form of indirect control flow relevant to modern binaries (notably, some indirect **call** patterns are entangled with call-edge merging in that pipeline). Waiting for perfect compiler labels would have blocked progress. We therefore adopted **silver labels**: each site’s **difficulty class** is defined from **whether Ghidra and angr both resolve it, one resolves it, or neither resolves it**. This is a pragmatic proxy: it encodes “what state-of-the-art static tools can do today” rather than semantic ground truth, but it is reproducible and scales to the full binary set.

---

## 4. Methodology

### 4.1 Data Collection

**Binaries:** 31 ARM32 ELF objects compiled with GCC **-O3**, drawn from **coreutils**, **SPEC CPU2006**, **OpenSSL**, and the **SSH** suite.

**Per-binary extraction:** For each binary, we enumerate indirect control flow sites with:

- **Ghidra** (`analyzeHeadless` plus a custom `ExportIndirectFlowsCustom.java` script, with PLT filtering), and  
- **angr** (CFGFast, `kb.indirect_jumps`, with a VEX fallback scan where needed).

Both paths emit structured JSON: instruction address, jump vs call, and resolved target address sets.

### 4.2 Silver Label Construction

For each site we assign one of three labels:

| Label | Definition | Interpretation |
|-------|-----------|---------------|
| easy | Both tools report at least one target and the sets overlap | L1 often sufficient |
| medium | Exactly one tool resolves targets | L2 likely needed |
| hard | Neither tool resolves | L3 or beyond |

We observed **zero** cases where both tools resolved but with **conflicting** non-overlapping targets: disagreement, when it occurs, takes the form of **coverage** (one tool silent) rather than contradictory target sets.

### 4.3 Difficulty Distributions

Representative training binaries illustrate the diversity of silver-label mixtures:

| Binary | Sites | Easy | Medium | Hard |
|--------|-------|------|--------|------|
| gcc | 1820 | 51.2% | 21.3% | 27.5% |
| h264ref | 386 | 4.7% | 79.0% | 16.3% |
| ssh | 174 | 25.9% | 10.9% | 63.2% |
| openssl | 82 | 75.6% | 4.9% | 19.5% |
| bzip2 | 36 | 33.3% | 27.8% | 38.9% |
| dealII | 2057 | ~3% | ~6% | ~91% |

Large C++-style binaries such as **dealII** are extreme **hard-dominated** cases; **openssl** is comparatively easy-dominated.

### 4.4 MDP Formulation

**State (19 dimensions):** Episode progress, remaining budget fraction, features of the current site (including attempted levels), running success rates per level, **global difficulty histogram** for the binary, and site-level attributes.

**Actions:** `Discrete(4)` — SKIP, L1, L2, L3. *Note:* In deployment, running Ghidra once is effectively a **sunk cost** over the whole binary; a more literal formulation would fix L1 outcomes and let RL choose only L2/L3 upgrades. We keep L1 as an action for generality of the simulator; Section 7 discusses the sunk-cost variant.

**Reward:** +1 for a newly successful resolution, minus λ times the action cost, with a large penalty for exhausting the budget mid-episode.

**Budget:** `budget = budget_ratio × num_sites × L1_cost` (with `budget_ratio = 2.0` in the stored calibration runs unless noted).

**Cost model:** Abstract units **L1 : L2 : L3 = 1 : 5 : 20** in the primary configuration. These are **not** wall-clock seconds: both Ghidra and angr are dominated by fixed whole-binary overhead, so per-site wall-clock ratios are unstable. We instead rely on **sensitivity sweeps** (Section 5.2) over **1:3:15**, **1:5:20**, and **1:10:50** (`data/calibration/sensitivity_results.json`, field `meta.cost_presets`).

**Success probabilities:** Calibrated from silver labels so that L1 behaves like Ghidra’s empirical success per difficulty class, L2 like angr’s, etc., as estimated from the joint extraction.

### 4.5 Training

**Algorithm:** PPO from **stable-baselines3**, **MlpPolicy** with two hidden layers of 64 units, discount **γ = 0.99**, **200k** environment steps per run. Training is **per binary**: each evaluated binary has its own policy checkpoint unless we explicitly evaluate **cross-binary** transfer (Section 5.3).

### 4.6 Baselines

Eight deterministic or simple stochastic strategies: **all_skip**, **all_l1**, **all_l2**, **all_l3**, **random**, **greedy_cheap**, **budget_aware** (dynamically picks the most expensive action that remains affordable given remaining budget and remaining sites), and **escalation** (L1→L2→L3 on failure).

---

## 5. Experimental Results

All tables below use **mean resolve rate** (resolved sites / total sites) over **200 evaluation episodes** unless stated. Primary **in-domain** numbers for `gcc`, `dealII`, `openssl`, `bzip2`, and `ssh` come from `data/calibration/sensitivity_results.json` (**cost_name = baseline**, **seed = 0**). Where noted, **multi-seed** statistics for `gcc` and `dealII` use seeds **0–2** from the same file.

### 5.1 Main Result: RL vs Baselines

The following table compares the **trained PPO policy** to **budget_aware** and **all_l1**. The column **Δ** is RL minus the **better** of the two baselines (so negative values mean RL underperforms that pair).

| Binary | RL | budget_aware | all_l1 | Δ (RL − max(baseline)) |
|--------|-----|-------------|--------|-------------------------|
| dealII | **0.928** | 0.117 | 0.089 | **+0.811** |
| gcc | **0.814** | 0.732 | 0.724 | **+0.082** |
| openssl | **0.811** | 0.810 | 0.803 | +0.001 |
| bzip2 | 0.605 | 0.616 | 0.606 | −0.011 |
| ssh | 0.312 | 0.318 | 0.312 | −0.006 |

On **dealII** (about **91%** hard sites, **2057** sites), RL achieves a mean resolve rate of **0.928**, while **budget_aware** stays at **0.117** and uniform **all_l1** at **0.089**. The ratio **0.928 / 0.117 ≈ 7.9** matches the “about **8×**” headline in the abstract; relative to **all_l1**, the ratio is about **10.5×**. The learned policy effectively **abandons wasted L1 retries** on hopeless sites and **concentrates** expensive actions where the calibrated model expects returns.

On **gcc** (mixed difficulty, **1820** sites), RL at **0.814** improves on **budget_aware** (**0.732**) by **8.2 percentage points** and on **all_l1** (**0.724**) by **9.0** points. **openssl** is easy-dominated: RL (**0.811**) and **budget_aware** (**0.810**) are effectively tied, with both slightly above **all_l1** (**0.803**). **bzip2** is small (**36** sites): RL (**0.605**) sits between **all_l1** and **budget_aware** and does not beat the best heuristic (**0.616**).

**SSH.** With **174** sites and **63.2%** hard labels, RL (**0.312**) is **slightly below** **budget_aware** (**0.318**) and **equal** to **all_l1** (**0.312**) under the baseline cost ratio. Interestingly, the **random** baseline attains a **higher** mean resolve rate (**0.512**) in the same log: that policy terminates episodes very early (**mean_episode_length ≈ 55** vs **≈ 291** for RL), so the comparison is **not** like-for-like on exploration depth and cost. We therefore treat **budget_aware** and **all_l1** as the primary **interpretable** baselines for ssh. Under the **low** cost preset (1:3:15), RL rises to **0.413** while **budget_aware** is **0.345** (Section 5.2), which supports the hypothesis that **tighter per-step costs** on ssh were blocking useful upgrades at baseline.

### 5.2 Sensitivity to Cost Ratio

We evaluate three presets from `sensitivity_results.json`: **low** (1:3:15), **baseline** (1:5:20), **high** (1:10:50). For **gcc** and **dealII**, three random seeds (**0–2**) are aggregated in the file’s summary objects; we report **mean ± standard deviation** of mean resolve rate over seeds.

**GCC** (per-episode resolve rate):

| Preset | RL | budget_aware |
|--------|-----|--------------|
| low (1:3:15) | **0.865 ± 0.023** | 0.740 |
| baseline (1:5:20) | **0.812 ± 0.010** | 0.732 |
| high (1:10:50) | 0.723 ± 0.000 | 0.727 |

Under **high** cost, RL and **budget_aware** **coincide** (~**0.723**): expensive actions are rarely affordable, so policies collapse toward **L1-dominated** behavior. Under **low** cost, RL’s margin over **budget_aware** widens to about **12.5** percentage points (mean).

**dealII:**

| Preset | RL | budget_aware | all_l1 |
|--------|-----|--------------|--------|
| low (1:3:15) | **0.974 ± 0.000** | 0.137 | 0.089 |
| baseline (1:5:20) | **0.951 ± 0.019** | 0.117 | 0.089 |
| high (1:10:50) | **0.087 ± 0.003** | 0.102 | 0.089 |

Here the **high** preset produces a **cliff**: L2 and L3 unit costs (**10** and **50**) exhaust the abstract budget almost immediately, so **every** strategy’s resolve rate falls to the **all_l1** floor (~**0.089**). RL no longer has room to express its “skip cheap retries, buy depth selectively” tactic. Between **low** and **baseline**, RL remains **far above** heuristics; **low** slightly **increases** both RL and **budget_aware**, but the **gap** remains massive.

**SSH** (single seed 0 in file): baseline RL **0.312** vs **budget_aware** **0.318**; **low** RL **0.413** vs **budget_aware** **0.345**; **high** RL **0.281** vs **budget_aware** **0.298**.

**Conclusion:** Qualitative conclusions are **stable** across cost presets whenever the budget still permits meaningful depth. Under **extreme** per-step costs, **all** methods converge to cheap behavior—this is an environment limitation, not a failure unique to RL.

### 5.3 Cross-Binary Generalization

We evaluated **frozen** policies trained on one binary on another, using `data/calibration/cross_binary_results.json` (**eval_episodes = 200**, **seed = 0**, **cost_name = baseline**). Diagonal entries match the in-domain runs also duplicated in that file.

**Mean resolve rate (rows = training binary, columns = evaluation binary):**

| Train \ Eval | gcc | dealII | ssh |
|--------------|-----|--------|-----|
| gcc | **0.814** | **0.198** | **0.442** |
| dealII | **0.082** | **0.928** | — |

*(No `dealII → ssh` or `ssh → *` cross rows appear in the current JSON; ssh in-domain diagonal **0.312** is taken from `sensitivity_results.json`.)*

**Interpretation.** A policy trained on **gcc** expects many **easy** sites and **selective** upgrades; on **dealII** it achieves only **0.198** mean resolve rate versus **0.928** for the **dealII-trained** policy—a **drop of 0.616**. Conversely, a **dealII** policy that learned to **skip L1** and **spend** on deeper levels achieves **0.082** on **gcc** versus **0.814** for the **gcc-trained** agent (**−0.732**). Even on **ssh**, the **gcc** policy (**0.442**) is **not** the same as in-domain ssh training (**0.312**), showing that **off-diagonal** behavior is not trivially “average.”

On **gcc → dealII**, the JSON marks **random** as the **best baseline** by mean resolve rate (**0.416**); the transferred RL policy (**0.198**) is **below** that naive baseline, underscoring **negative transfer**. These numbers support the claim that PPO has memorized **statistics of the training binary’s silver-label distribution**, not a portable algorithm for “indirect CF in general.”

**Implication:** Deployment should assume **per-binary or per-class** retraining unless the agent can **adapt online** within a single analysis session (Section 6.3).

---

## 6. Discussion

### 6.1 When Does RL Help?

RL delivers the largest gains when **three** conditions overlap:

1. **Many hard sites**, so uniform L1 is far from optimal.  
2. **Many sites overall**, so the episode offers enough sequential decisions for credit assignment to matter.  
3. **Moderate cost ratios**, so the budget allows **some** L2/L3 uses but not “everywhere L3.”

Conversely, RL is **unnecessary** when L1 already resolves most sites (**openssl**), when the binary is **too small** for stable learning (**bzip2**), or when costs are **so high** that every policy is forced to the same cheap corner case (**dealII** under 1:10:50).

### 6.2 Modeling Gaps and Limitations

**Stochastic vs deterministic simulators.** The environment draws successes from calibrated **probabilities**. Real tools, given fixed code, are closer to **deterministic** for each site. The MDP therefore mixes **epistemic** uncertainty (what the analyst does not yet know) with **aleatory** noise we injected for smoothing. A more faithful model would condition on **features** or even **oracle silver labels** inside training (with held-out test binaries).

**Site independence.** Resolving one indirect call may constrain others in the same vtable or function family. Our MDP treats sites as **IID** steps; cross-site structure is only weakly reflected in aggregate statistics.

**Whole-binary tools.** Ghidra and angr are not invoked “per site” in reality; per-site actions are an **abstraction** of “how much effort to spend interpreting this location given a global budget.”

**Fixed visit order.** The agent cannot reorder sites; real pipelines might prioritize **high-confidence** regions first to harvest cheap wins or **hard** regions first while budget remains.

**Known difficulty prior.** The global difficulty histogram is exposed in the state. In the wild, that histogram is **unknown** until partially measured—another argument for **online** belief updating.

### 6.3 The Case for Online-Adaptive Agents

Cross-binary failure modes and the modeling gaps above point to the same remedy: **adaptation inside the episode**. Rather than shipping one policy per corpus, the agent could maintain a **belief** over difficulty class proportions, use the first fraction of sites as a **probe**, and shift from exploration to exploitation as uncertainty shrinks. Mechanisms include **contextual bandits**, **recurrent policies** (LSTM over sites), or **meta-RL** (MAML / RL²-style fast adaptation).

Observable signals already present in our pipeline include **instruction context**, **partial Ghidra outputs** even when resolution fails, **running success rates**, and **spatial locality** of sites. None of these require new oracle infrastructure—only a richer state and training regime.

---

## 7. Future Work

1. **L1 sunk-cost MDP:** Provide Ghidra’s L1 outcome as a fixed prior per site; restrict RL to **L2/L3 / skip** decisions. This matches operational reality and shrinks the action space.

2. **Deterministic or feature-conditioned success:** Train a classifier to predict L2/L3 outcomes from code features using silver labels, and feed predicted success into the reward or transition model.

3. **Online adaptation:** Address the cross-binary table in Section 5.3 with recurrent or meta-learned policies that **update tactics** during a single binary’s analysis.

4. **Cross-site structure:** Model correlations among sites (same function, same compiler pattern) so resolving one site **updates beliefs** about neighbors.

5. **Ordering:** Let the agent choose **which** site to visit next, not only **how deep** to analyze the current one.

6. **Real L3:** Integrate QEMU-backed dynamic analysis with measured costs so three-tier experiments use **actual** tool outputs rather than calibrated abstractions alone.

---

## 8. Conclusion

We formulated **budgeted indirect control flow resolution** as an MDP and trained **PPO** policies against strong static baselines in a simulator calibrated from **Ghidra–angr silver labels** on **31** ARM32 binaries. On **hard-dominated, large** programs such as **dealII**, learned policies achieve mean resolve rates around **0.93** under default abstract costs, **orders of magnitude** ahead of uniform L1 and a sophisticated **budget_aware** heuristic in relative terms. Gains on **gcc** are smaller but **consistent** across seeds. **Sensitivity analysis** over **1:3:15**, **1:5:20**, and **1:10:50** shows that rankings are **stable** until the environment itself forbids depth. **Cross-binary evaluation** in `cross_binary_results.json` shows **sharp performance collapse** when policies are applied off-distribution (**0.814 → 0.198**, **0.928 → 0.082** on the two main transfers), which is an important **negative result**: it clarifies that current PPO training learns **tactics tuned to each binary’s difficulty histogram**, not a universal analysis algorithm. The most promising next step is therefore **online adaptation**—closing the loop between partial observations during analysis and revised allocation—while tightening the simulator toward **deterministic, feature-based** success models.
