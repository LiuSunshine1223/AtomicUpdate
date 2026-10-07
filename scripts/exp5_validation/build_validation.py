"""Build and freeze experiment 5. Reads model metadata; makes zero generation calls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atomic_memory import paths
from atomic_memory.backend import OllamaBackend
from atomic_memory import validation as v


def protocol():
    return """# Controlled endpoint validation (experiment 5)

This protocol is frozen before any new reader response. The main experiment's
1,200 trials and Y/Z contrasts remain separate and unchanged.

## Design and construction

50 independently generated base cases (seed 20261013); two subjects, six facts,
one target update selected by core.generate_cases(..., targets_per_case=1).
The shared fact order is generated before target selection. Two fixed g3 units,
m00/m01, contain three assertions each in the clean reference S*. The target
subject's three attributes are queried. All six variants use the same S* oracle:

1. clean: six canonical, correctly updated current assertions.
2. surface: replace only the target sentence by "At present, {subject}'s
   {attribute label} is {value}." Five other sentences and memberships unchanged.
3. contradiction: insert the old target value as a current assertion immediately
   after the new assertion in its unit; keep the new and other assertions.
4. deletion: remove the target assertion; keep both unit IDs and all other facts.
5. misbinding: change only the target assertion's subject to the other subject.
   Keep its value, attribute, unit and position; disclose other-subject competition.
6. other_damage: choose the (case_index mod 3)-th other-subject assertion in shared
   order and replace its value by the next member of its fixed vocabulary (cyclic).

The surface template is designed to preserve the same positive current-state
proposition at a common evaluation time. Shared tuples alone do not establish
equivalence for arbitrary natural-language renderings. There are no free
paraphrases, temporal shifts, pronouns or qualifiers. Some interventions change
several strict-check components; they are not isolated manipulations of one flag.

All prior retained case batches, integration smoke, and deterministic removed
development/feasibility batches are excluded by identical initial-fact-set hash.
No cases are selected according to reader results. All 50 clean states are kept,
including any reader failures. No updater or additional reader is run.

## Independent state truth and endpoints

Structured current assertions and construction operations supply state labels;
the strict parser and reader outputs do not supply these labels. Completeness
means each queried attribute has an assertion. Uniqueness is false with competing
distinct values, true with three singleton value sets, and null when incomplete
without competition. Identical duplicate assertions do not create distinct
values, but global-reference equality compares multisets (duplicates matter).
Oracle support means each S* profile value is explicitly present for the queried
subject. Support does not imply uniqueness. Counts, unit membership and competing
slots are recorded independently. Misbinding can introduce an unrequested slot
or a competing other-subject value; both are retained.

Existing strict Y uses the original canonical grammar, six reference facts,
counts and fixed semantic-slot membership. Action legality/execution are held
true: this is a state-content/representation-contract study, not updater success.
Z is exact equality to S* for all variants. A reasonable UNKNOWN after deletion
still fails oracle-profile recovery; abstention is reported separately.
Actual answer support compares returned concrete values with constructed current
values for that subject and attribute. UNKNOWN is null (undetermined support).
Any unsupported concrete field makes whole-answer support false; otherwise
invalid responses or abstention have null support. No human/LLM annotation is used.

## Reader execution and freeze

Use the qualified exp4 Llama 3.1 8B Instruct Q4_K_M model, prompt and JSON schema;
temperature 0, seed 42, output limit 64, context 8192. Read-only metadata preflight
must match the qualified exp4 model/runtime. Exact source/code/artifact hashes
and actual model digest/runtime are in freeze_manifest.json.
300 logical evaluations in a seeded shuffled schedule (20261014). Cache only
identical complete system/user prompts, schema and fixed generation configuration
within this batch, bound to the frozen model digest. Score each logical row
against its own oracle. Never reuse older responses as new observations.
Save every raw response, including invalid/unexpected answers. Transport errors
stop the run; resume only unfinished inputs. No result-driven retries, exclusions,
replacement readers or changes to templates, oracles or analysis.

## Analysis and limitations

Report all six arms (n=50 each), strict components, JSON validity, abstention,
unsupported concrete answers, answer support and Y/Z cross-tabs. Pair all 50
clean-to-intervention outcomes, including clean failures. The case is the cluster;
300 variants/cached outputs are not independent cases. Percentile 95% intervals
use 20,000 whole-case bootstrap resamples (numpy RandomState seed 20261015), with
all six variants retained together. No significance tests or new primary endpoint.
The fixed within-input generation seed does not constitute repeated sampling.

One answer per input cannot establish within-input instability or order effects.
This g3-only study illustrates constructed disagreement pathways; it cannot
identify all causes of g1-minus-g3 effect changes, retro-certify the original
53 generated disagreements, validate arbitrary natural language, prove reader
independence or estimate deployment prevalence. Main contrast magnitudes come
from the original paired experiment. Stop after this fixed batch, offline
verification and one manuscript update; do not tune according to results.
"""


def build(out=v.OUT, backend=None):
    out = Path(out)
    paths.require_output_stage(out, Path(__file__))
    if (out / "freeze_manifest.json").exists():
        freeze, plan = v.verify(out)
        print("VERIFIED_EXISTING | {} cases | {} logical inputs | {} unique calls".format(
            v.N_CASES, len(plan), freeze["unique_inputs"]))
        return freeze
    require_files = {"cases.jsonl", "reader_plan.jsonl", "protocol.md"}
    if out.exists():
        v.require(all(p.name in require_files or p.name.endswith(".tmp") for p in out.iterdir()),
                  "Unfrozen files present; preserve this directory before rebuilding")
    context = v.source_context(paths.OUTPUT_ROOT)
    cases, plan, overlap = v.build_design(context)
    backend = backend or OllamaBackend(paths.OLLAMA_URL, timeout_seconds=30)
    snapshot = v.normalize_snapshot(backend.describe(v.CONFIG["model"]))
    v.require(snapshot == context["qualified_snapshot"],
              "Model/runtime differs from the passed exp4 reader controls. "
              "This batch requires that qualified snapshot; no generations were made.")
    v.write_jsonl_atomic(out / "cases.jsonl", cases)
    v.write_jsonl_atomic(out / "reader_plan.jsonl", plan)
    v.write_text_atomic(out / "protocol.md", protocol())
    artifacts = {name: v.digest(out / name) for name in sorted(require_files)}
    freeze = dict(protocol_version=v.VERSION, frozen_at_utc=v.now(),
                  num_cases=v.N_CASES, logical_rows=len(plan),
                  unique_inputs=len({r["input_sha256"] for r in plan}),
                  variants=list(v.VARIANTS), cluster_unit="case",
                  case_seed=v.CASE_SEED, schedule_seed=v.SCHEDULE_SEED,
                  bootstrap_seed=v.BOOTSTRAP_SEED, bootstrap_replicates=v.BOOTSTRAP_REPLICATES,
                  model_snapshot=snapshot, generation_config=context["config"],
                  source_sha256={name: v.digest(paths.OUTPUT_ROOT / name) for name in v.SOURCE_FILES},
                  code_sha256={name: v.digest(paths.PROJECT_ROOT / name) for name in v.CODE_FILES},
                  artifact_sha256=artifacts, overlap_audit=overlap,
                  construction_counts={arm: {"rows": sum(r["variant"] == arm for r in plan),
                      "strict_pass": sum(r["strict_evaluation"]["end_to_end_success"]
                                         for r in plan if r["variant"] == arm)} for arm in v.VARIANTS},
                  reader_calls_at_freeze=0, older_responses_reused=False)
    v.write_json_atomic(out / "freeze_manifest.json", freeze)
    v.verify(out)
    print("FROZEN | 50 independent cases x 6 variants = 300 logical rows")
    print("Overlap: 0 | generation calls: 0 | pending unique calls: {}".format(freeze["unique_inputs"]))
    return freeze


if __name__ == "__main__":
    build()
