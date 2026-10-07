# Profile Extraction Layer-2 — Frozen Offline Analysis

This report is generated offline from the frozen prospective formal batch. It makes **zero model calls**.

## Integrity

- Formal cases: **50**; two targets per case.
- Logical updater/downstream rows: **1,200/1,200**.
- Cluster unit: **case**; both targets remain inside each resampled case.
- Whole-case bootstrap: **20,000** replicates.
- Updater models are analyzed separately; no pooled model-average estimand is used.
- Failure-reason labels overlap; their row counts are not additive.
- Failure-class/task relations are descriptive and do not establish causality.

## Headline state–function result

- Strict state success: **1050 / 1200 (87.5%)**.
- Functional task success: **1103 / 1200 (91.9%)**.
- Strict-fail / task-pass: **53 / 150 strict failures (35.3%)**.
- Strict-pass / task-fail: **0**.

These are observed joint outcomes under the fixed Llama Profile Extraction reader. They do not imply that strict evaluation is unnecessary or that a strict failure is harmless for other downstream tasks.

## Condition-level strict vs functional success

| Updater | Condition | Strict | Strict 95% case-bootstrap CI | Functional | Functional 95% case-bootstrap CI | Strict-fail / Task-pass |
|---|---|---:|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | g1 / Select | 100/100 (100.0%) | [100.0% , 100.0%] | 100/100 (100.0%) | [100.0% , 100.0%] | 0 |
| llama3.1:8b-instruct-q4_K_M | g1 / Patch | 100/100 (100.0%) | [100.0% , 100.0%] | 100/100 (100.0%) | [100.0% , 100.0%] | 0 |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | 64/100 (64.0%) | [55.0% , 73.0%] | 78/100 (78.0%) | [71.0% , 85.0%] | 14 |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | 74/100 (74.0%) | [66.0% , 82.0%] | 75/100 (75.0%) | [67.0% , 83.0%] | 1 |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | 98/100 (98.0%) | [95.0% , 100.0%] | 98/100 (98.0%) | [95.0% , 100.0%] | 0 |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | 97/100 (97.0%) | [93.0% , 100.0%] | 97/100 (97.0%) | [93.0% , 100.0%] | 0 |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | 82/100 (82.0%) | [75.0% , 88.0%] | 93/100 (93.0%) | [88.0% , 97.0%] | 11 |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | 82/100 (82.0%) | [74.0% , 89.0%] | 82/100 (82.0%) | [74.0% , 89.0%] | 0 |
| gemma3:12b-it-q4_K_M | g1 / Select | 100/100 (100.0%) | [100.0% , 100.0%] | 100/100 (100.0%) | [100.0% , 100.0%] | 0 |
| gemma3:12b-it-q4_K_M | g1 / Patch | 94/100 (94.0%) | [89.0% , 98.0%] | 99/100 (99.0%) | [97.0% , 100.0%] | 5 |
| gemma3:12b-it-q4_K_M | g3 / Select | 90/100 (90.0%) | [84.0% , 96.0%] | 93/100 (93.0%) | [87.0% , 98.0%] | 3 |
| gemma3:12b-it-q4_K_M | g3 / Patch | 69/100 (69.0%) | [61.0% , 77.0%] | 88/100 (88.0%) | [82.0% , 94.0%] | 19 |

## Frozen g1 − g3 contrasts

| Updater | Contrast | Role | Estimate | 95% whole-case bootstrap CI |
|---|---|---|---:|---:|
| llama3.1:8b-instruct-q4_K_M | F_Patch | PRIMARY_FUNCTIONAL | +25.0 pp | [+17.0 pp, +33.0 pp] |
| llama3.1:8b-instruct-q4_K_M | G_Patch | MATCHING_STATE | +26.0 pp | [+18.0 pp, +34.0 pp] |
| llama3.1:8b-instruct-q4_K_M | F_Select | SUPPORTING_FUNCTIONAL | +22.0 pp | [+15.0 pp, +30.0 pp] |
| llama3.1:8b-instruct-q4_K_M | G_Select | SUPPORTING_STATE | +36.0 pp | [+27.0 pp, +45.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | F_Patch | PRIMARY_FUNCTIONAL | +15.0 pp | [+7.0 pp, +24.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | G_Patch | MATCHING_STATE | +15.0 pp | [+7.0 pp, +24.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | F_Select | SUPPORTING_FUNCTIONAL | +5.0 pp | [+0.0 pp, +10.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | G_Select | SUPPORTING_STATE | +16.0 pp | [+9.0 pp, +23.0 pp] |
| gemma3:12b-it-q4_K_M | F_Patch | PRIMARY_FUNCTIONAL | +11.0 pp | [+5.0 pp, +17.0 pp] |
| gemma3:12b-it-q4_K_M | G_Patch | MATCHING_STATE | +25.0 pp | [+17.0 pp, +33.0 pp] |
| gemma3:12b-it-q4_K_M | F_Select | SUPPORTING_FUNCTIONAL | +7.0 pp | [+2.0 pp, +13.0 pp] |
| gemma3:12b-it-q4_K_M | G_Select | SUPPORTING_STATE | +10.0 pp | [+4.0 pp, +17.0 pp] |

`F_*` uses functional task success; `G_*` uses strict state success. Positive values mean the observed g1 rate is higher than g3 for that interface. These are model-specific paired contrasts, not model rankings.

## Failure-class functional audit

Failure-reason summaries are descriptive because labels overlap. The CSV outputs retain event count, distinct-case count, task-pass count, task-fail count, and pass proportion for every observed model/condition/reason combination.

| Updater | Condition | Failure reason | Rows | Cases | Task pass | Task fail | Functional pass rate |
|---|---|---|---:|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | g3 / Select | locator_failure | 13 | 13 | 5 | 8 | 38.5% |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | protected_fact_loss | 30 | 28 | 9 | 21 | 30.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | stale_old_retained | 13 | 13 | 5 | 8 | 38.5% |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | structure_failure | 33 | 28 | 12 | 21 | 36.4% |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | target_new_missing | 1 | 1 | 0 | 1 | 0.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Select | unsupported_change | 3 | 3 | 2 | 1 | 66.7% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | execution_failure | 24 | 23 | 0 | 24 | 0.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | locator_failure | 20 | 19 | 0 | 20 | 0.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | protected_fact_loss | 1 | 1 | 0 | 1 | 0.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | stale_old_retained | 25 | 23 | 0 | 25 | 0.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | structure_failure | 1 | 1 | 1 | 0 | 100.0% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | target_new_missing | 26 | 24 | 1 | 25 | 3.8% |
| llama3.1:8b-instruct-q4_K_M | g3 / Patch | unsupported_change | 2 | 2 | 1 | 1 | 50.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | locator_failure | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | protected_fact_loss | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | stale_old_retained | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | structure_failure | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | target_new_missing | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Select | unsupported_change | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | execution_failure | 2 | 2 | 0 | 2 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | locator_failure | 2 | 2 | 0 | 2 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | stale_old_retained | 2 | 2 | 0 | 2 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | target_new_missing | 3 | 3 | 0 | 3 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g1 / Patch | unsupported_change | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | locator_failure | 12 | 12 | 8 | 4 | 66.7% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | protected_fact_loss | 10 | 10 | 3 | 7 | 30.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | stale_old_retained | 13 | 13 | 8 | 5 | 61.5% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | structure_failure | 18 | 18 | 11 | 7 | 61.1% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Select | unsupported_change | 3 | 3 | 3 | 0 | 100.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | execution_failure | 17 | 15 | 0 | 17 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | locator_failure | 16 | 15 | 0 | 16 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | protected_fact_loss | 1 | 1 | 0 | 1 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | stale_old_retained | 18 | 16 | 0 | 18 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | target_new_missing | 18 | 16 | 0 | 18 | 0.0% |
| qwen2.5:7b-instruct-q4_K_M | g3 / Patch | unsupported_change | 1 | 1 | 0 | 1 | 0.0% |
| gemma3:12b-it-q4_K_M | g1 / Patch | execution_failure | 1 | 1 | 0 | 1 | 0.0% |
| gemma3:12b-it-q4_K_M | g1 / Patch | locator_failure | 1 | 1 | 0 | 1 | 0.0% |
| gemma3:12b-it-q4_K_M | g1 / Patch | memory_contract_failure | 5 | 5 | 5 | 0 | 100.0% |
| gemma3:12b-it-q4_K_M | g1 / Patch | stale_old_retained | 1 | 1 | 0 | 1 | 0.0% |
| gemma3:12b-it-q4_K_M | g1 / Patch | target_new_missing | 1 | 1 | 0 | 1 | 0.0% |
| gemma3:12b-it-q4_K_M | g3 / Select | locator_failure | 5 | 4 | 1 | 4 | 20.0% |
| gemma3:12b-it-q4_K_M | g3 / Select | memory_contract_failure | 3 | 3 | 2 | 1 | 66.7% |
| gemma3:12b-it-q4_K_M | g3 / Select | stale_old_retained | 7 | 6 | 1 | 6 | 14.3% |
| gemma3:12b-it-q4_K_M | g3 / Select | structure_failure | 4 | 4 | 1 | 3 | 25.0% |
| gemma3:12b-it-q4_K_M | g3 / Select | target_new_missing | 3 | 3 | 0 | 3 | 0.0% |
| gemma3:12b-it-q4_K_M | g3 / Patch | execution_failure | 11 | 11 | 0 | 11 | 0.0% |
| gemma3:12b-it-q4_K_M | g3 / Patch | locator_failure | 4 | 4 | 0 | 4 | 0.0% |
| gemma3:12b-it-q4_K_M | g3 / Patch | memory_contract_failure | 19 | 18 | 19 | 0 | 100.0% |
| gemma3:12b-it-q4_K_M | g3 / Patch | stale_old_retained | 12 | 12 | 0 | 12 | 0.0% |
| gemma3:12b-it-q4_K_M | g3 / Patch | target_new_missing | 12 | 12 | 0 | 12 | 0.0% |

## Interpretation guardrails

- `strict-fail / task-pass` is state–function disagreement for this fixed local Profile Extraction task, not proof that the underlying memory state is globally correct.
- `strict-fail / task-fail` is an observed co-occurrence, not proof that the strict failure caused the downstream failure.
- Functional outcomes are conditional on the fixed reader and target-subject three-attribute scope.
- Corruption outside the target subject can be functionally irrelevant to this task while still being a strict-state failure.
- The study uses natural-rate failures under this controlled synthetic regime; it does not estimate deployment prevalence.
- g1 vs g3 is an endpoint comparison only; no monotonicity claim is made.

## Output files

- `condition_summary.csv` — condition rates and case-bootstrap CIs.
- `contrast_summary.csv` — frozen F/G contrasts and whole-case bootstrap CIs.
- `state_function_gap_summary.csv` — descriptive Task − Strict gaps.
- `joint_outcome_summary.csv` — four-way Strict × Task counts.
- `failure_reason_functional_summary.csv` — overlapping failure reasons × task outcomes.
- `failure_stage_functional_summary.csv` — mutually exclusive primary failure stage × task outcomes.
- `strict_failure_profile_class_summary.csv` — downstream profile classes among strict failures.

