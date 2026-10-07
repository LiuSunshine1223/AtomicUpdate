# Layer-2 Reader-Control Offline Error Audit

**Mode:** ZERO MODEL CALLS  
**Purpose:** diagnose the failed frozen reader-suitability gate; do not tune the gate from these results.

## 1. Control outcomes

| Control | Correct/expected | n | Rate |
|---|---:|---:|---:|
| Oracle g1 | 88 | 100 | 88.0% |
| Oracle g3 | 86 | 100 | 86.0% |
| Stale g1 | 92 | 100 | 92.0% |
| Stale g3 | 88 | 100 | 88.0% |
| No-memory gold | 11 | 100 | 11.0% |

## 2. Are failures shared across g1 and g3?

### Oracle

| Pattern | Tasks |
|---|---:|
| both_expected | 83 |
| g1_only_expected | 5 |
| g3_only_expected | 3 |
| neither_expected | 9 |

Oracle failure overlap:

- fail both groupings: **9**
- g1-only failure: **3**
- g3-only failure: **5**

### Stale

| Pattern | Tasks |
|---|---:|
| both_expected | 86 |
| g1_only_expected | 6 |
| g3_only_expected | 2 |
| neither_expected | 6 |

Stale failure overlap:

- fail both groupings: **6**
- g1-only failure: **2**
- g3-only failure: **6**

## 3. What wrong profile does the reader choose?

### oracle__g1

| Selected semantic role | Count |
|---|---:|
| stale_target | 5 |
| protected_alt:city | 2 |
| protected_alt:drink | 2 |
| protected_alt:device | 1 |
| protected_alt:hobby | 1 |
| protected_alt:job | 1 |

### oracle__g3

| Selected semantic role | Count |
|---|---:|
| stale_target | 5 |
| protected_alt:drink | 3 |
| protected_alt:city | 2 |
| protected_alt:device | 2 |
| protected_alt:hobby | 1 |
| protected_alt:job | 1 |

### stale__g1

| Selected semantic role | Count |
|---|---:|
| gold | 3 |
| multi_alt:job,drink | 3 |
| multi_alt:city,device | 1 |
| multi_alt:device,hobby | 1 |

### stale__g3

| Selected semantic role | Count |
|---|---:|
| gold | 4 |
| multi_alt:job,drink | 4 |
| multi_alt:city,device | 2 |
| multi_alt:city,hobby | 1 |
| multi_alt:city,job,drink | 1 |

## 4. No-memory position behavior

- Gold hits: **11/100 (11.0%)**

| Selected profile | Count |
|---|---:|
| p0 | 47 |
| p1 | 5 |
| p2 | 1 |
| p3 | 1 |
| p6 | 8 |
| p7 | 38 |

## 5. Accuracy by target attribute

| Arm/grouping | Attribute | n | Gold acc. | Stale-expected rate |
|---|---|---:|---:|---:|
| no_memory__None | city | 28 | 14.3% | 7.1% |
| no_memory__None | device | 18 | 22.2% | 11.1% |
| no_memory__None | drink | 16 | 6.2% | 12.5% |
| no_memory__None | hobby | 18 | 5.6% | 5.6% |
| no_memory__None | job | 20 | 5.0% | 15.0% |
| oracle__g1 | city | 28 | 85.7% | 3.6% |
| oracle__g1 | device | 18 | 94.4% | 0.0% |
| oracle__g1 | drink | 16 | 75.0% | 12.5% |
| oracle__g1 | hobby | 18 | 94.4% | 5.6% |
| oracle__g1 | job | 20 | 90.0% | 5.0% |
| oracle__g3 | city | 28 | 78.6% | 7.1% |
| oracle__g3 | device | 18 | 94.4% | 0.0% |
| oracle__g3 | drink | 16 | 75.0% | 18.8% |
| oracle__g3 | hobby | 18 | 94.4% | 0.0% |
| oracle__g3 | job | 20 | 90.0% | 0.0% |
| stale__g1 | city | 28 | 3.6% | 92.9% |
| stale__g1 | device | 18 | 0.0% | 94.4% |
| stale__g1 | drink | 16 | 6.2% | 93.8% |
| stale__g1 | hobby | 18 | 0.0% | 100.0% |
| stale__g1 | job | 20 | 5.0% | 80.0% |
| stale__g3 | city | 28 | 3.6% | 85.7% |
| stale__g3 | device | 18 | 11.1% | 88.9% |
| stale__g3 | drink | 16 | 0.0% | 93.8% |
| stale__g3 | hobby | 18 | 0.0% | 100.0% |
| stale__g3 | job | 20 | 5.0% | 75.0% |

## 6. Interpretation guardrails

- This audit is descriptive. It must not be used to lower the frozen reader gate after observing failures.
- Shared failures across g1 and g3 suggest task/reader difficulty; grouping-discordant failures suggest reader sensitivity to memory layout. Neither pattern alone establishes a causal mechanism.
- No prospective updater calls should begin while the frozen reader-suitability gate remains failed.

## 7. Error-row inventory

Detailed unexpected Oracle/Stale rows are saved in `reader_control_error_cases.jsonl`.

Total unexpected Oracle/Stale rows: **46**.
