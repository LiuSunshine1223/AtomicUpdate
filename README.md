# AtomicUpdate

Code and experimental records for **Evaluating LLM Memory Updates: State Checks and Downstream Profile Accuracy**.

The project studies how fact grouping and edit interfaces affect LLM memory updates, and how strict state checks and downstream profile matching change estimated configuration effects. Manuscript LaTeX sources are maintained separately.

## Study design

Each memory contains six facts about two fictional subjects, with three attributes per subject. An update changes one current fact while the remaining facts and unit structure are checked.

- **Strict state endpoint (Y):** action legality, execution success, target correctness, preservation of protected facts, sentence grammar, and memory structure. The constructed-state study holds legality and execution successful and evaluates the state-content and representation-contract components.
- **Profile endpoint (Z):** exact matching of the queried subject's three returned attributes against the reference profile.
- **Explanatory audits:** queried-state completeness, competing current values, answer support, reader sensitivity, and presentation-order sensitivity.

## Repository layout

| Path | Contents |
| --- | --- |
| atomic_memory/ | Case construction, update execution, Ollama backend, reader interface, audits, paths, and offline verification |
| scripts/ | Experiment builders, model runners, and analyses |
| outputs/ | Cases, plans, responses, scores, analyses, protocols, and record manifests |
| plot/ | Figure-generation code and shared styling |
| figure/ | Code-generated figures and their manuscript layout variants |
| schematic/ (when needed) | Paper images created outside the plotting code, such as manually drawn architecture diagrams; retain editable sources and exports together |
| tests/ | Deterministic tests of execution, grouping, paths, runtime guards, audits, and constructed interventions |
| requirements.txt | Python dependencies |
| LICENSE | MIT license |

`atomic_memory/reader.py` defines the shared prompts, response schemas, parsing, and scoring. `backend.py` handles Ollama requests and response metadata. Model identities, generation settings, input hashes, and software versions are recorded by batch.

## Reproduce saved results

Run from the project root with Python 3.9+. Install the dependencies, then verify the saved records. Offline verification does not require Ollama or model weights.

```powershell
python -m pip install -r requirements.txt
Remove-Item Env:ATOMICUPDATE_OUTPUT_ROOT -ErrorAction SilentlyContinue

python -B -m unittest discover -s tests -p "test_*.py"
python -B scripts/exp1_baseline/analyze_baseline.py --check
python -B scripts/exp2_wording/analyze_wording.py --check
python -B scripts/exp3_unit_load/analyze_unit_load.py --check
python -B scripts/exp4_profile/analyze_profile.py --check
python -B scripts/exp4_profile/analyze_profile_audit.py --check
python -B scripts/exp4_profile/followup/reader_followup.py analyze --check
python -B scripts/exp5_validation/analyze_validation.py --check
python -B scripts/exp5_validation/order_check.py --check
```

On Linux or macOS, use `unset ATOMICUPDATE_OUTPUT_ROOT` in place of `Remove-Item`; the Python commands are identical. `--check` recomputes analyses in memory and verifies saved values, counts, and input hashes without model calls or file writes. The final analysis summary reports `files_written: 0`.

## Experiments and execution stages

The same experiment directory names are used under `scripts/` and `outputs/`.

| Directory | Study | Size |
| --- | --- | --- |
| exp1_baseline/ | Fact representation and edit-scope baseline | 20 cases x 2 targets x 5 conditions x 3 models; 600 updater calls |
| exp2_wording/ | Two instruction paraphrases on the baseline cases | 1,200 additional updater calls |
| exp3_unit_load/ | Groupings of 1, 2, or 3 facts per unit | 20 independent cases; 720 updater calls |
| exp4_profile/ | Strict and profile endpoints on the same states | 50 independent cases; 1,200 updater calls and 1,200 logical readouts |
| exp4_profile/followup/ | Query audit, second reader, order probes, and controls | 2,060 logical rows; 605 additional unique responses |
| exp5_validation/ | Clean, surface wording, contradiction, deletion, misbinding, and other-subject damage | 50 independent cases x 6 variants; 300 reader calls |
| exp5_validation/order_check/ | Adjacent new/old assertion swap | 50 pairs from experiment 5; 50 additional reader calls |
| preparation/ | Integration gate, candidate-selection controls, and extraction calibration | dev, selection, and calibration evidence |

The four updater studies contain 3,720 calls over 90 distinct cases; experiment 2 reuses experiment 1's cases. Constructed validation and the adjacent-swap check contain 350 reader calls over the same 50 constructed cases. Batches retain their own protocols and denominators.

The integration gate precedes the baseline. Candidate-selection controls and direct-extraction calibration establish the reader interface; the main profile study then uses fresh cases. Its stage order is build, reader controls, updater execution, profile readout, and analysis. The follow-up and constructed-state stages each require their own freeze before new observations.

Each target and condition starts from its own copy of the original memory. Whole-case bootstrap resampling keeps both targets and repeated conditions together. Reader caching reduces physical calls while retaining all logical rows. Failure labels can overlap; cost summaries describe tokens and generation latency, with no retrieval-cost measurement.

## Experimental records

Records are provided under `outputs/`. Transport metadata uses the generic local loopback address `http://127.0.0.1:11434` for publication. Corresponding protocol fingerprints and dependent file-hash references were recomputed for the distributed records. Model inputs, raw responses, scores, model digests, generation settings, and experimental counts are preserved.

The main profile study contains 1,050/1,200 strict passes and 1,103/1,200 primary-reader exact matches, including 53 strict-fail / exact-match rows. The query audit covers all 1,200 states; 27 are not fully scorable under the canonical parser. Grammar relaxation is reported separately as post-hoc sensitivity analysis.

The second reader scores 94/100 on the stale-g3 control, below its 95/100 gate, so its analyses remain exploratory. Constructed contradictions yield 6/50 exact matches in New-to-Old order and 42/50 after an adjacent Old-to-New swap; both states remain contradictory. Exact matching does not establish global correctness or unique, noncontradictory support. Findings are limited to the recorded synthetic tasks, models, reader, and constructions.

## Figures

The data flow is `scripts/ -> outputs/ -> plot/ -> figure/`. Plot values come from saved analyses and raw records.

| Plotting script | Data source under outputs/ | PDFs under figure/ |
| --- | --- | --- |
| plot_baseline.py | Baseline and wording analyses | exp1_baseline/: framework, main_results, paired_contrasts, failure_profiles, inference_overhead |
| plot_wording.py | Baseline and wording analyses | exp2_wording/: wording_sensitivity |
| plot_unit_load.py | exp3_unit_load/unit_load_analysis/ | exp3_unit_load/: unit_load_results |
| plot_profile.py | exp4_profile/formal_analysis.json | exp4_profile/: profile_framework, profile_contrasts |
| plot_reader.py | exp4_profile/followup/analysis.json | exp4_profile/followup/: reader_robustness |
| plot_validation.py | Constructed validation and order-check analyses | exp5_validation/: endpoint_validation, endpoint_validation_manuscript |

Filenames omit `.pdf`. The main manuscript uses profile_framework, profile_contrasts, and endpoint_validation; the supplement uses reader_robustness. `plot/plot_validation.py` generates both validation layouts by default. Use `--only analysis` or `--only manuscript` to select one. The compact layout is retained as `figure/exp5_validation/endpoint_validation_manuscript.pdf`. With `--paper-root`, the selected manuscript layout is copied to `figures/endpoint_validation.pdf`; when both layouts are generated, only the compact layout is copied. Both layouts use the same verified records and experimental counts.

Both framework diagrams are generated by plotting code and belong under `figure/`. Create `schematic/` when adding paper images made outside that code, preserving files such as `.drawio`, `.svg`, or `.pptx` alongside their PDF exports. No such assets are currently included.

Regenerate figures in a separate directory:

```powershell
python -B plot/plot_baseline.py --output-dir figure_check/exp1_baseline
python -B plot/plot_wording.py --output-dir figure_check/exp2_wording
python -B plot/plot_unit_load.py --output-dir figure_check/exp3_unit_load
python -B plot/plot_profile.py --output-dir figure_check/exp4_profile
python -B plot/plot_reader.py --output-dir figure_check/exp4_profile/followup
python -B plot/plot_validation.py --output-dir figure_check/exp5_validation
```

These commands make no model calls. Omitting `--output-dir` writes to the corresponding `figure/` directory. `--paper-root` copies generated PDFs into a manuscript's `figures/` directory. PDF creation metadata can vary between runs.

## Models and configuration

Weights are managed separately by local Ollama. Model tags are:

- `llama3.1:8b-instruct-q4_K_M`
- `qwen2.5:7b-instruct-q4_K_M`
- `gemma3:12b-it-q4_K_M`

Consult batch manifests for checkpoint digests, Ollama versions, prompts, schemas, and generation settings. Matching tags do not guarantee matching digests, and fixed seeds do not guarantee identical outputs across environments.

`ATOMICUPDATE_OLLAMA_URL` selects the service; the default is the generic local loopback service. `ATOMICUPDATE_OUTPUT_ROOT` selects an independent output root for a new batch. Keep machine-specific environment configuration outside version control; `.env*` files are ignored.

New model batches require matching preparation evidence, qualification gates, and a new freeze. Existing write-stage guards reject conflicting inputs, source versions, or configuration. Use the offline checks for completed observations and separate output directories for new experiments.

## Code and record versions

For a fixed manuscript version, cite the repository commit SHA or its version tag. Batch manifests under `outputs/` record the source hashes, model digests, generation settings, and software versions for each experiment; use the corresponding manifest when matching the original run environment.

## License

The repository uses the [MIT License](LICENSE), with `Copyright (c) 2026 Xiangyang Liu`. It includes research code, documentation, and accompanying experimental records. Model weights remain subject to their publishers' licenses.
