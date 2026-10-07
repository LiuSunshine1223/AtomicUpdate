# Unit-load extension analysis

This is the frozen offline analysis of the 720-call extension. Models are analyzed separately. Confidence intervals resample whole cases and preserve both targets and all six conditions.

## End-to-end success

| Model | L1 Select | L1 Patch | L2 Select | L2 Patch | L3 Select | L3 Patch |
|---|---:|---:|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | 100.0% | 97.5% | 70.0% | 82.5% | 65.0% | 82.5% |
| qwen2.5:7b-instruct-q4_K_M | 97.5% | 100.0% | 90.0% | 92.5% | 75.0% | 87.5% |
| gemma3:12b-it-q4_K_M | 100.0% | 100.0% | 95.0% | 95.0% | 95.0% | 90.0% |

## Frozen primary and key secondary contrasts

| Model | Contrast | Estimate | 95% case-cluster bootstrap interval |
|---|---|---:|---:|
| llama3.1:8b-instruct-q4_K_M | G_Patch | +15.0 pp | [+2.5 pp, +27.5 pp] |
| llama3.1:8b-instruct-q4_K_M | S | +20.0 pp | [+10.0 pp, +30.0 pp] |
| qwen2.5:7b-instruct-q4_K_M | G_Patch | +12.5 pp | [+0.0 pp, +27.5 pp] |
| qwen2.5:7b-instruct-q4_K_M | S | +10.0 pp | [+0.0 pp, +20.0 pp] |
| gemma3:12b-it-q4_K_M | G_Patch | +10.0 pp | [+2.5 pp, +20.0 pp] |
| gemma3:12b-it-q4_K_M | S | -5.0 pp | [-15.0 pp, +5.0 pp] |

Interpretation guardrails:

- `G_Patch > 0` means finer grouping had higher observed Patch success in this sample; it does not by itself establish a deployment recommendation.
- `S > 0` only means the load-1 versus load-3 grouping contrast is more positive under Select than Patch. It does not establish Patch as the better absolute configuration.
- Similar observed success rates do not establish equivalence or noninferiority.
- Grouping load changes unit size and unit count together, so these are configuration effects rather than a pure causal effect of abstract granularity.

## Token overhead

| Model | Condition | Mean input | Mean output | Mean total | Warm latency (s) |
|---|---|---:|---:|---:|---:|
| llama3.1:8b-instruct-q4_K_M | Load 1 / Select | 444.4 | 33.8 | 478.2 | 0.293 |
| llama3.1:8b-instruct-q4_K_M | Load 1 / Patch | 445.4 | 37.5 | 482.9 | 0.285 |
| llama3.1:8b-instruct-q4_K_M | Load 2 / Select | 387.5 | 40.8 | 428.4 | 0.335 |
| llama3.1:8b-instruct-q4_K_M | Load 2 / Patch | 388.5 | 38.0 | 426.5 | 0.287 |
| llama3.1:8b-instruct-q4_K_M | Load 3 / Select | 368.7 | 46.8 | 415.5 | 0.366 |
| llama3.1:8b-instruct-q4_K_M | Load 3 / Patch | 369.7 | 38.2 | 408.0 | 0.287 |
| qwen2.5:7b-instruct-q4_K_M | Load 1 / Select | 448.4 | 34.8 | 483.2 | 0.271 |
| qwen2.5:7b-instruct-q4_K_M | Load 1 / Patch | 449.4 | 38.8 | 488.2 | 0.291 |
| qwen2.5:7b-instruct-q4_K_M | Load 2 / Select | 388.5 | 42.3 | 430.8 | 0.321 |
| qwen2.5:7b-instruct-q4_K_M | Load 2 / Patch | 389.5 | 42.4 | 431.9 | 0.316 |
| qwen2.5:7b-instruct-q4_K_M | Load 3 / Select | 368.7 | 49.5 | 418.2 | 0.357 |
| qwen2.5:7b-instruct-q4_K_M | Load 3 / Patch | 369.7 | 44.4 | 414.1 | 0.332 |
| gemma3:12b-it-q4_K_M | Load 1 / Select | 503.0 | 38.9 | 541.9 | 0.644 |
| gemma3:12b-it-q4_K_M | Load 1 / Patch | 505.0 | 46.4 | 551.3 | 0.711 |
| gemma3:12b-it-q4_K_M | Load 2 / Select | 430.8 | 48.5 | 479.3 | 0.806 |
| gemma3:12b-it-q4_K_M | Load 2 / Patch | 432.8 | 51.7 | 484.5 | 0.759 |
| gemma3:12b-it-q4_K_M | Load 3 / Select | 406.5 | 58.7 | 465.2 | 0.846 |
| gemma3:12b-it-q4_K_M | Load 3 / Patch | 408.5 | 54.8 | 463.3 | 0.781 |

Token counts are inference-overhead measurements within a model, not storage, retrieval, monetary, or utility estimates. Latency is descriptive for the recorded backend.

## Analysis status

The primary and key secondary estimands, bootstrap unit, number of repetitions, and analysis seed were frozen before the 720 formal calls. Supporting contrasts are reported to preserve the six-cell absolute context and must not be used as a post-hoc significance screen.
