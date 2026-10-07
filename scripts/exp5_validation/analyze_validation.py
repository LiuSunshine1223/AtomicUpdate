"""Offline paired analysis of experiment 5; --check writes nothing and uses no network."""
from pathlib import Path
import csv
import io
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from atomic_memory import validation as v
from atomic_memory.reanalysis import run_analysis
from atomic_memory.validation import write_json_atomic, write_text_atomic

STRICT_FIELDS = (
    "legal_action", "execution_success", "contract_valid", "structure_valid",
    "membership_valid", "target_new_current", "target_old_not_current",
    "protected_facts_preserved", "no_unsupported_changes",
)
SUMMARY_COLUMNS = (
    "variant", "n", "strict_pass", "exact_match", "exact_match_ci_low_pp",
    "exact_match_ci_high_pp", "valid_json", "any_unknown", "all_unknown",
    "supported_answer", "unsupported_concrete_answer", "support_undetermined",
    "unsupported_exact_match", "contradictory_supported_exact_match", "Y0_Z1",
)
PAIRED_COLUMNS = (
    "variant", "n", "strict_diff_pp", "profile_diff_pp", "profile_diff_ci_low_pp",
    "profile_diff_ci_high_pp", "endpoint_change_pp", "endpoint_change_ci_low_pp",
    "endpoint_change_ci_high_pp", "Y_1_1", "Y_1_0", "Y_0_1", "Y_0_0",
    "Z_1_1", "Z_1_0", "Z_0_1", "Z_0_0",
)


def write_csv_atomic(path, rows, fieldnames):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomic(path, stream.getvalue())


def ci(values):
    return [float(x) for x in np.percentile(values, [2.5, 97.5])]


def summarize(results):
    lookup = {(r["case_id"], r["variant"]): r for r in results}
    case_ids = sorted({r["case_id"] for r in results})
    v.require(len(case_ids) == v.N_CASES and len(lookup) == len(results) == v.N_CASES * len(v.VARIANTS),
              "Expected 50 complete, unique six-variant case clusters")
    v.require(set(lookup) == {(c, a) for c in case_ids for a in v.VARIANTS}, "Missing paired variant")
    y = np.array([[lookup[c, a]["strict_evaluation"]["end_to_end_success"]
                   for a in v.VARIANTS] for c in case_ids], dtype=float)
    z = np.array([[lookup[c, a]["profile_exact_match"] for a in v.VARIANTS]
                  for c in case_ids], dtype=float)
    # One resampling index matrix keeps the entire six-arm case together.
    index = np.random.RandomState(v.BOOTSTRAP_SEED).randint(
        len(case_ids), size=(v.BOOTSTRAP_REPLICATES, len(case_ids)))
    z_boot = z[index].mean(axis=1) * 100
    y_boot = y[index].mean(axis=1) * 100
    arms, pairs = [], []
    for j, arm in enumerate(v.VARIANTS):
        rows = [lookup[c, arm] for c in case_ids]
        low, high = ci(z_boot[:, j])
        count = lambda key: sum(bool(r[key]) for r in rows)
        strict_components = {field: {
            "true": sum(r["strict_evaluation"][field] is True for r in rows),
            "false": sum(r["strict_evaluation"][field] is False for r in rows),
            "unscored": sum(r["strict_evaluation"][field] is None for r in rows),
        } for field in STRICT_FIELDS}
        failures = sorted({reason for r in rows for reason in r["strict_evaluation"]["failure_reasons"]})
        failure_counts = {reason: sum(reason in r["strict_evaluation"]["failure_reasons"] for r in rows)
                          for reason in failures}
        yz = {"Y{}_Z{}".format(a, b): int(((y[:, j] == a) & (z[:, j] == b)).sum())
              for a in (0, 1) for b in (0, 1)}
        support_counts = {"supported_answer": sum(r["answer_supported"] is True for r in rows),
                          "unsupported_concrete_answer": sum(r["answer_supported"] is False for r in rows),
                          "support_undetermined": sum(r["answer_supported"] is None for r in rows)}
        arms.append(dict(variant=arm, n=len(rows), strict_pass=int(y[:, j].sum()),
            exact_match=int(z[:, j].sum()), strict_rate_pp=float(y[:, j].mean() * 100),
            exact_match_rate_pp=float(z[:, j].mean() * 100), exact_match_ci_low_pp=low,
            exact_match_ci_high_pp=high, valid_json=count("valid_json"),
            any_unknown=count("any_unknown"), all_unknown=count("all_unknown"), **support_counts,
            unsupported_exact_match=sum(r["answer_supported"] is False and r["profile_exact_match"] for r in rows),
            contradictory_supported_exact_match=sum(r["profile_exact_match"] and r["answer_supported"] is True
                and r["state_properties"]["queried_profile_unique_and_noncontradictory"] is False for r in rows),
            strict_components=strict_components, strict_failure_counts=failure_counts,
            queried_profile_complete=sum(r["state_properties"]["queried_profile_complete"] for r in rows),
            queried_profile_unique=sum(r["state_properties"]["queried_profile_unique_and_noncontradictory"] is True for r in rows),
            oracle_profile_supported=sum(r["state_properties"]["oracle_profile_supported"] for r in rows),
            global_reference_exact=sum(r["state_properties"]["global_reference_exact"] for r in rows),
            **yz))
        if j:
            delta_z, delta_y = z[:, j] - z[:, 0], y[:, j] - y[:, 0]
            low, high = ci(z_boot[:, j] - z_boot[:, 0])
            gap_low, gap_high = ci((z_boot[:, j] - z_boot[:, 0]) - (y_boot[:, j] - y_boot[:, 0]))
            transitions = {"{}_{}_{}".format(metric, a, b): int(((matrix[:, 0] == a) & (matrix[:, j] == b)).sum())
                           for metric, matrix in (("Y", y), ("Z", z)) for a in (0, 1) for b in (0, 1)}
            pairs.append(dict(variant=arm, n=len(rows), strict_diff_pp=float(delta_y.mean() * 100),
                profile_diff_pp=float(delta_z.mean() * 100), profile_diff_ci_low_pp=low,
                profile_diff_ci_high_pp=high, endpoint_change_pp=float((delta_z - delta_y).mean() * 100),
                endpoint_change_ci_low_pp=gap_low, endpoint_change_ci_high_pp=gap_high, **transitions))
    return dict(cluster_unit="case", num_cases=len(case_ids), logical_rows=len(results),
                interval_method="whole-case paired percentile bootstrap, 95%",
                bootstrap_replicates=v.BOOTSTRAP_REPLICATES, bootstrap_seed=v.BOOTSTRAP_SEED,
                denominator_note="All counts use all 50 cases per arm; false and unscored are separate.",
                contrast_direction="intervention minus clean; endpoint change = profile difference minus strict difference",
                scope="Fixed g3 constructed states, qualified primary reader; exploratory endpoint validation",
                arm_summary=arms, paired_transitions=pairs)


def analyze(out=v.OUT):
    out = Path(out)
    freeze, plan = v.verify(out, check_code=False)
    response_path, result_path = out / "responses.jsonl", out / "reader_results.jsonl"
    v.require(response_path.is_file() and result_path.is_file(), "Run run_validation.py to completion first")
    cache = v.validate_raw(v.records(response_path), plan, freeze["model_snapshot"], complete=True)
    results = v.score(plan, cache)
    v.require(results == v.records(result_path), "Saved scores differ from independent raw-response rescoring")
    summary = v.load(out / "run_summary.json")
    v.require(summary["status"] == "COMPLETE" and summary["logical_rows"] == len(plan)
              and summary["unique_responses"] == len(cache), "Incomplete saved run summary")
    for key, name in (("freeze_manifest_sha256", "freeze_manifest.json"),
                      ("responses_sha256", "responses.jsonl"), ("reader_results_sha256", "reader_results.jsonl")):
        v.require(summary[key] == v.digest(out / name), "Run summary mismatch: " + name)
    data = summarize(results)
    data.update(protocol_version=freeze["protocol_version"], created_at_utc=v.now(),
                analysis_script_sha256=v.digest(Path(__file__)), unique_responses=len(cache),
                within_batch_cache_reuses=len(plan) - len(cache), model_snapshot=freeze["model_snapshot"],
                source_hashes={name: v.digest(out / name) for name in
                              ("freeze_manifest.json", "responses.jsonl", "reader_results.jsonl", "protocol.md")})
    write_json_atomic(out / "analysis.json", data)
    write_csv_atomic(out / "arm_summary.csv", data["arm_summary"], SUMMARY_COLUMNS)
    write_csv_atomic(out / "paired_transitions.csv", data["paired_transitions"], PAIRED_COLUMNS)
    print("ANALYZED | 50 case clusters | all 300 logical rows | 20,000 paired bootstrap resamples")


if __name__ == "__main__":
    run_analysis(analyze, globals(), v.OUT, Path(__file__))
