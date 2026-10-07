# Instruction-Only Robustness — Paired Offline Sensitivity Analysis

This analysis reuses the same 20 formal cases. It is a sensitivity analysis, not an independent validation.

Only system-instruction wording differs across the baseline and two pre-specified paraphrases. Memory, evidence, condition definitions, user-prompt adapter, schemas, evaluator, models, and generation settings remain frozen.

## E2E success across instruction wordings

### llama3.1:8b-instruct-q4_K_M

| Condition | Baseline | Paraphrase 1 | Paraphrase 2 |
|---|---:|---:|---:|
| A_compound_global | 33/40 (82.5%) | 31/40 (77.5%) | 36/40 (90.0%) |
| B_compound_select | 28/40 (70.0%) | 31/40 (77.5%) | 21/40 (52.5%) |
| C_atomic_global | 39/40 (97.5%) | 40/40 (100.0%) | 40/40 (100.0%) |
| D_atomic_select | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) |
| E_compound_exact_patch | 35/40 (87.5%) | 34/40 (85.0%) | 34/40 (85.0%) |

### qwen2.5:7b-instruct-q4_K_M

| Condition | Baseline | Paraphrase 1 | Paraphrase 2 |
|---|---:|---:|---:|
| A_compound_global | 34/40 (85.0%) | 35/40 (87.5%) | 33/40 (82.5%) |
| B_compound_select | 37/40 (92.5%) | 35/40 (87.5%) | 35/40 (87.5%) |
| C_atomic_global | 32/40 (80.0%) | 37/40 (92.5%) | 34/40 (85.0%) |
| D_atomic_select | 39/40 (97.5%) | 39/40 (97.5%) | 39/40 (97.5%) |
| E_compound_exact_patch | 39/40 (97.5%) | 39/40 (97.5%) | 39/40 (97.5%) |

### gemma3:12b-it-q4_K_M

| Condition | Baseline | Paraphrase 1 | Paraphrase 2 |
|---|---:|---:|---:|
| A_compound_global | 39/40 (97.5%) | 40/40 (100.0%) | 39/40 (97.5%) |
| B_compound_select | 39/40 (97.5%) | 40/40 (100.0%) | 40/40 (100.0%) |
| C_atomic_global | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) |
| D_atomic_select | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) |
| E_compound_exact_patch | 40/40 (100.0%) | 40/40 (100.0%) | 40/40 (100.0%) |

## Paired change relative to the baseline wording

Differences are paraphrase minus baseline on the exact same case/update/condition/model. Confidence intervals resample whole cases.

| Model | Wording | Condition | E2E delta | 95% case-cluster CI | Improved / Worsened / Tied targets |
|---|---|---|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | A_compound_global | -5.0 pp | [-15.0 pp, +5.0 pp] | 1 / 3 / 36 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | B_compound_select | +7.5 pp | [-2.5 pp, +17.5 pp] | 4 / 1 / 35 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | C_atomic_global | +2.5 pp | [+0.0 pp, +7.5 pp] | 1 / 0 / 39 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | E_compound_exact_patch | -2.5 pp | [-7.5 pp, +0.0 pp] | 0 / 1 / 39 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | A_compound_global | +7.5 pp | [-7.5 pp, +22.5 pp] | 6 / 3 / 31 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | B_compound_select | -17.5 pp | [-30.0 pp, -5.0 pp] | 1 / 8 / 31 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | C_atomic_global | +2.5 pp | [+0.0 pp, +7.5 pp] | 1 / 0 / 39 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | E_compound_exact_patch | -2.5 pp | [-7.5 pp, +0.0 pp] | 1 / 2 / 37 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | A_compound_global | +2.5 pp | [-10.0 pp, +15.0 pp] | 4 / 3 / 33 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | B_compound_select | -5.0 pp | [-12.5 pp, +0.0 pp] | 0 / 2 / 38 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | C_atomic_global | +12.5 pp | [+0.0 pp, +25.0 pp] | 6 / 1 / 33 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | E_compound_exact_patch | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | A_compound_global | -2.5 pp | [-12.5 pp, +7.5 pp] | 2 / 3 / 35 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | B_compound_select | -5.0 pp | [-12.5 pp, +0.0 pp] | 0 / 2 / 38 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | C_atomic_global | +5.0 pp | [-7.5 pp, +17.5 pp] | 4 / 2 / 34 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | E_compound_exact_patch | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | A_compound_global | +2.5 pp | [+0.0 pp, +7.5 pp] | 1 / 0 / 39 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | B_compound_select | +2.5 pp | [+0.0 pp, +7.5 pp] | 1 / 0 / 39 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | C_atomic_global | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | E_compound_exact_patch | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | A_compound_global | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | B_compound_select | +2.5 pp | [+0.0 pp, +7.5 pp] | 1 / 0 / 39 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | C_atomic_global | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | D_atomic_select | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | E_compound_exact_patch | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |

## Wording range

The range is max minus min E2E rate across the three wordings. It is descriptive sensitivity, not an independent uncertainty estimate.

| Model | Condition | Baseline | P1 | P2 | Max-min range |
|---|---|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | A_compound_global | 82.5% | 77.5% | 90.0% | +12.5 pp |
| llama3.1:8b-instruct-q4_K_M | B_compound_select | 70.0% | 77.5% | 52.5% | +25.0 pp |
| llama3.1:8b-instruct-q4_K_M | C_atomic_global | 97.5% | 100.0% | 100.0% | +2.5 pp |
| llama3.1:8b-instruct-q4_K_M | D_atomic_select | 100.0% | 100.0% | 100.0% | +0.0 pp |
| llama3.1:8b-instruct-q4_K_M | E_compound_exact_patch | 87.5% | 85.0% | 85.0% | +2.5 pp |
| qwen2.5:7b-instruct-q4_K_M | A_compound_global | 85.0% | 87.5% | 82.5% | +5.0 pp |
| qwen2.5:7b-instruct-q4_K_M | B_compound_select | 92.5% | 87.5% | 87.5% | +5.0 pp |
| qwen2.5:7b-instruct-q4_K_M | C_atomic_global | 80.0% | 92.5% | 85.0% | +12.5 pp |
| qwen2.5:7b-instruct-q4_K_M | D_atomic_select | 97.5% | 97.5% | 97.5% | +0.0 pp |
| qwen2.5:7b-instruct-q4_K_M | E_compound_exact_patch | 97.5% | 97.5% | 97.5% | +0.0 pp |
| gemma3:12b-it-q4_K_M | A_compound_global | 97.5% | 100.0% | 97.5% | +2.5 pp |
| gemma3:12b-it-q4_K_M | B_compound_select | 97.5% | 100.0% | 100.0% | +2.5 pp |
| gemma3:12b-it-q4_K_M | C_atomic_global | 100.0% | 100.0% | 100.0% | +0.0 pp |
| gemma3:12b-it-q4_K_M | D_atomic_select | 100.0% | 100.0% | 100.0% | +0.0 pp |
| gemma3:12b-it-q4_K_M | E_compound_exact_patch | 100.0% | 100.0% | 100.0% | +0.0 pp |

## Representation x edit-scope interaction by wording

Interaction = `(D-B)-(C-A)`.

| Model | Wording | Interaction | 95% case-cluster CI |
|---|---|---:|---:|
| llama3.1:8b-instruct-q4_K_M | baseline | +15.0 pp | [-5.0 pp, +35.0 pp] |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | +0.0 pp | [-15.0 pp, +12.5 pp] |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | +37.5 pp | [+25.0 pp, +52.5 pp] |
| qwen2.5:7b-instruct-q4_K_M | baseline | +10.0 pp | [+0.0 pp, +20.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | +5.0 pp | [-7.5 pp, +17.5 pp] |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | +7.5 pp | [-7.5 pp, +22.5 pp] |
| gemma3:12b-it-q4_K_M | baseline | +0.0 pp | [-7.5 pp, +7.5 pp] |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | +0.0 pp | [+0.0 pp, +0.0 pp] |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | -2.5 pp | [-7.5 pp, +0.0 pp] |

## Atomic Select versus Compound Exact Patch

Difference = D Atomic Select minus E Compound Exact Patch.

| Model | Wording | D-E difference | 95% case-cluster CI | D-only / E-only / Tie |
|---|---|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | baseline | +12.5 pp | [+5.0 pp, +22.5 pp] | 5 / 0 / 35 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_1 | +15.0 pp | [+5.0 pp, +25.0 pp] | 6 / 0 / 34 |
| llama3.1:8b-instruct-q4_K_M | instruction_paraphrase_2 | +15.0 pp | [+5.0 pp, +25.0 pp] | 6 / 0 / 34 |
| qwen2.5:7b-instruct-q4_K_M | baseline | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_1 | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| qwen2.5:7b-instruct-q4_K_M | instruction_paraphrase_2 | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | baseline | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_1 | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |
| gemma3:12b-it-q4_K_M | instruction_paraphrase_2 | +0.0 pp | [+0.0 pp, +0.0 pp] | 0 / 0 / 40 |

## Interpretation guardrails

- This uses the same 20 cases, so it measures instruction-wording sensitivity rather than independent replication.
- Do not declare a configuration robust merely because its rank stays the same; compare paired effect magnitudes and clustered intervals.
- Large wording shifts should first be treated as instability of the design conclusion, not as a positive contribution by themselves.
- Exact Patch remains a core comparator; similar observed rates do not prove equivalence.
- Interaction estimates must be reported regardless of significance if the paper centers representation x edit scope.

