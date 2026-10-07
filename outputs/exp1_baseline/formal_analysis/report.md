# Representation × Edit-Scope — Frozen Formal Pilot Analysis

This report is generated entirely offline from the frozen 600-call formal pilot. It makes no model calls.

## Integrity audit

- Formal cases: **20**; two update targets per case.
- Formal calls: **600/600**, with 600 unique result keys and planned call indices 1–600 complete.
- Formal preflight: **PASS**; frozen protocol manifest verified.
- Statistical clustering unit: **case**; the two targets within one case remain clustered.
- Failure-reason labels are multi-label and overlapping; counts are not additive.
- Cost analysis covers input tokens, output tokens, and generation latency. **Retrieval cost is not measured.**

## End-to-end success by model and condition

| Model | A Compound Global | B Compound Select | C Atomic Global | D Atomic Select | E Exact Patch |
|---|---:|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | 33 / 40 (82.5%) | 28 / 40 (70.0%) | 39 / 40 (97.5%) | 40 / 40 (100.0%) | 35 / 40 (87.5%) |
| qwen2.5:7b-instruct-q4_K_M | 34 / 40 (85.0%) | 37 / 40 (92.5%) | 32 / 40 (80.0%) | 39 / 40 (97.5%) | 39 / 40 (97.5%) |
| gemma3:12b-it-q4_K_M | 39 / 40 (97.5%) | 39 / 40 (97.5%) | 40 / 40 (100.0%) | 40 / 40 (100.0%) | 40 / 40 (100.0%) |

Cluster-bootstrap 95% confidence intervals for these rates are stored in `condition_summary.csv`.

## Preplanned paired contrasts

Effects are **right condition minus left condition** in E2E success rate. Positive and negative values are descriptive effect directions, not model rankings.

| Model | Contrast | Difference | 95% case-cluster bootstrap CI | Right-only / Left-only / Tie | Exploratory case sign-flip p |
|---|---|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | C_minus_A | 0.150 | [0.025, 0.300] | 7 / 1 / 32 | 0.1094 |
| llama3.1:8b-instruct-q4_K_M | D_minus_B | 0.300 | [0.150, 0.450] | 12 / 0 / 28 | 0.0020 |
| llama3.1:8b-instruct-q4_K_M | B_minus_A | -0.125 | [-0.325, 0.075] | 4 / 9 / 27 | 0.3408 |
| llama3.1:8b-instruct-q4_K_M | D_minus_C | 0.025 | [0.000, 0.075] | 1 / 0 / 39 | 1.0000 |
| llama3.1:8b-instruct-q4_K_M | E_minus_B | 0.175 | [0.075, 0.275] | 7 / 0 / 33 | 0.0156 |
| qwen2.5:7b-instruct-q4_K_M | C_minus_A | -0.050 | [-0.150, 0.050] | 1 / 3 / 36 | 0.6250 |
| qwen2.5:7b-instruct-q4_K_M | D_minus_B | 0.050 | [0.000, 0.125] | 2 / 0 / 38 | 0.5000 |
| qwen2.5:7b-instruct-q4_K_M | B_minus_A | 0.075 | [-0.025, 0.175] | 5 / 2 / 33 | 0.3750 |
| qwen2.5:7b-instruct-q4_K_M | D_minus_C | 0.175 | [0.075, 0.275] | 7 / 0 / 33 | 0.0156 |
| qwen2.5:7b-instruct-q4_K_M | E_minus_B | 0.050 | [0.000, 0.125] | 2 / 0 / 38 | 0.5000 |
| gemma3:12b-it-q4_K_M | C_minus_A | 0.025 | [0.000, 0.075] | 1 / 0 / 39 | 1.0000 |
| gemma3:12b-it-q4_K_M | D_minus_B | 0.025 | [0.000, 0.075] | 1 / 0 / 39 | 1.0000 |
| gemma3:12b-it-q4_K_M | B_minus_A | 0.000 | [-0.075, 0.075] | 1 / 1 / 38 | 1.0000 |
| gemma3:12b-it-q4_K_M | D_minus_C | 0.000 | [0.000, 0.000] | 0 / 0 / 40 | 1.0000 |
| gemma3:12b-it-q4_K_M | E_minus_B | 0.025 | [0.000, 0.075] | 1 / 0 / 39 | 1.0000 |

The sign-flip p-value is an exploratory paired case-level diagnostic; the effect size and clustered confidence interval should remain the main reported quantities.

## 2 x 2 Representation x Edit-Scope interaction

For A-D, the descriptive interaction is defined as `(D-B)-(C-A)`: the Atomic-minus-Compound difference under Select-unit editing minus the same representation difference under Global rewrite.

| Model | Interaction | 95% case-cluster bootstrap CI | Positive / Negative / Zero target interactions |
|---|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | 0.150 | [-0.050, 0.350] | 10 / 4 / 26 |
| qwen2.5:7b-instruct-q4_K_M | 0.100 | [0.000, 0.225] | 5 / 1 / 34 |
| gemma3:12b-it-q4_K_M | 0.000 | [-0.075, 0.075] | 1 / 1 / 38 |

These interaction estimates are descriptive under 20 independent case clusters. A positive point estimate means the Atomic-minus-Compound difference is larger under Select-unit editing than under Global rewrite. It should not be described as a stable interaction when the case-cluster bootstrap interval includes zero.

## Locator success where localization is required

| Model | B Compound Select | D Atomic Select | E Exact Patch |
|---|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | 85.0% (n=40) | 100.0% (n=40) | 87.5% (n=40) |
| qwen2.5:7b-instruct-q4_K_M | 100.0% (n=40) | 100.0% (n=40) | 100.0% (n=40) |
| gemma3:12b-it-q4_K_M | 100.0% (n=40) | 100.0% (n=40) | 100.0% (n=40) |

## Cost / overhead

Input/output token counts are overhead measures. They should not be described as retrieval cost.

| Model | Condition | Mean input tokens | Mean output tokens | Raw mean latency (s) | Warm mean latency (s) |
|---|---|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | A_compound_global | 384.5 | 121.5 | 1.687 | 0.984 |
| llama3.1:8b-instruct-q4_K_M | B_compound_select | 387.5 | 41.2 | 0.363 | 0.363 |
| llama3.1:8b-instruct-q4_K_M | C_atomic_global | 441.3 | 178.4 | 1.482 | 1.482 |
| llama3.1:8b-instruct-q4_K_M | D_atomic_select | 444.3 | 33.8 | 0.310 | 0.310 |
| llama3.1:8b-instruct-q4_K_M | E_compound_exact_patch | 388.5 | 38.6 | 0.357 | 0.357 |
| qwen2.5:7b-instruct-q4_K_M | A_compound_global | 385.5 | 126.7 | 1.241 | 0.901 |
| qwen2.5:7b-instruct-q4_K_M | B_compound_select | 388.5 | 42.2 | 0.304 | 0.304 |
| qwen2.5:7b-instruct-q4_K_M | C_atomic_global | 445.3 | 186.4 | 1.322 | 1.322 |
| qwen2.5:7b-instruct-q4_K_M | D_atomic_select | 448.3 | 34.8 | 0.257 | 0.257 |
| qwen2.5:7b-instruct-q4_K_M | E_compound_exact_patch | 389.5 | 43.5 | 0.348 | 0.348 |
| gemma3:12b-it-q4_K_M | A_compound_global | 427.4 | 149.7 | 2.461 | 1.980 |
| gemma3:12b-it-q4_K_M | B_compound_select | 430.4 | 48.6 | 0.743 | 0.743 |
| gemma3:12b-it-q4_K_M | C_atomic_global | 499.4 | 221.6 | 2.834 | 2.834 |
| gemma3:12b-it-q4_K_M | D_atomic_select | 502.4 | 38.5 | 0.647 | 0.647 |
| gemma3:12b-it-q4_K_M | E_compound_exact_patch | 432.4 | 51.6 | 0.828 | 0.828 |

Warm latency excludes the first formal generation call for each model because that call can contain a model-loading/cold-start penalty.

## Multi-label failure diagnostics

Across all 600 rows, the number of rows carrying each diagnostic label is:

- `structure_failure`: **31** rows
- `protected_fact_loss`: **15** rows
- `target_new_missing`: **15** rows
- `locator_failure`: **11** rows
- `stale_old_retained`: **11** rows
- `unsupported_change`: **8** rows
- `execution_failure`: **5** rows
- `memory_contract_failure`: **5** rows

These counts overlap. For example, one failed update can carry both `locator_failure` and `stale_old_retained`; therefore the counts must not be summed to infer a total number of failures.

## Reporting guardrails

- Analyze the three model families separately as the primary view; do not let a pooled average hide model heterogeneity.
- Do not tune prompts, regenerate formal cases, or rerun failed outputs after seeing these results.
- The formal study supports controlled synthetic evidence about representation granularity and edit scope; it does not by itself establish universal superiority of Atomic memory.

