"""Post-hoc Bonferroni sensitivity for the three primary Patch contrasts.

Reuse the original case bootstrap stream; leave all existing outputs unchanged.
No model calls are made.
"""
from pathlib import Path
from collections import defaultdict
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from atomic_memory import paths
from scripts.exp4_profile import analyze_profile as analysis


def summarize():
    saved = analysis.read_json(paths.PROFILE / "formal_analysis.json")
    for filename, expected in saved["source_hashes"].items():
        if analysis.sha256_file(paths.PROFILE / filename) != expected:
            raise ValueError("Formal source hash mismatch: " + filename)
    rows = analysis.read_jsonl(paths.PROFILE / "actual_reader_results.jsonl")
    if len(rows) != 1200:
        raise ValueError("Expected the complete 1,200-row primary-reader record")
    cells = defaultdict(dict)
    for row in rows:
        key = (row["updater_model"], row["condition"], row["case_id"])
        target = row["update_id"]
        if target in cells[key]:
            raise ValueError("Duplicate target identity")
        if row["functional_task_pass"] != (row["parsed_response"] == row["oracle_expected_values"]):
            raise ValueError("Profile score disagrees with the logged exact match")
        cells[key][target] = row
    case_ids = sorted({r["case_id"] for r in rows})
    if len(case_ids) != 50 or len(cells) != 600 or any(len(v) != 2 for v in cells.values()):
        raise ValueError("Incomplete case/configuration pairing")
    nominal = {(r["model"], r["contrast_id"]): r for r in saved["contrast_summary"]}
    rng = random.Random(analysis.BOOTSTRAP_SEED + 1)
    family_size = 3
    adjusted_level = 1 - .05 / family_size
    result = []
    # Iterate all original contrasts, including non-primary draws, to reproduce
    # the RNG state used for each updater's F_Patch contrast.
    for model in analysis.MODELS:
        for spec in analysis.CONTRAST_SPECS:
            values = []
            for case in case_ids:
                left = cells[model, spec["left"], case]
                right = cells[model, spec["right"], case]
                if set(left) != set(right):
                    raise ValueError("Grouping conditions have different targets")
                values.append(analysis.mean([
                    float(left[t][spec["metric"]]) - float(right[t][spec["metric"]])
                    for t in sorted(left)
                ]))
            draws = [sum(values[rng.randrange(50)] for _ in range(50)) / 50
                     for _ in range(analysis.BOOTSTRAP_REPS)]
            low, high = analysis.percentile_ci(draws)
            point = 100 * analysis.mean(values)
            ref = nominal[model, spec["contrast_id"]]
            for actual, expected in [(point, ref["estimate_percentage_points"]),
                                     (100 * low, ref["cluster_bootstrap_ci_low_pp"]),
                                     (100 * high, ref["cluster_bootstrap_ci_high_pp"])]:
                if abs(actual - expected) > 1e-9:
                    raise ValueError("Original bootstrap estimate or interval changed")
            if spec["contrast_id"] != "F_Patch":
                continue
            adj_low, adj_high = analysis.percentile_ci(draws, level=adjusted_level)
            result.append({"model": model, "contrast_id": "F_Patch",
                           "estimate_pp": point, "nominal_ci_low_pp": 100 * low,
                           "nominal_ci_high_pp": 100 * high,
                           "bonferroni_ci_low_pp": 100 * adj_low,
                           "bonferroni_ci_high_pp": 100 * adj_high})
    return {"analysis_role": "POST_HOC_MULTIPLICITY_SENSITIVITY",
            "family": "F_Patch for the three fixed updaters", "family_size": family_size,
            "nominal_familywise_level": .95, "per_interval_level": adjusted_level,
            "cluster_unit": "case", "case_clusters": 50, "targets_per_case": 2,
            "bootstrap_replicates": analysis.BOOTSTRAP_REPS,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED + 1,
            "bootstrap_stream": "Original model/contrast loop, including non-primary draws",
            "model_calls": 0,
            "interpretation": "Nominal Bonferroni percentile-bootstrap sensitivity; coverage is approximate",
            "source_hashes": {"formal_analysis.json": analysis.sha256_file(paths.PROFILE / "formal_analysis.json"),
                              "actual_reader_results.jsonl": analysis.sha256_file(paths.PROFILE / "actual_reader_results.jsonl")},
            "contrasts": result}


def main():
    result = summarize()
    destination = paths.PROFILE / "primary_multiplicity.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in result["contrasts"]:
        print("{}: {:+g} pp [{:+g}, {:+g}]".format(
            row["model"], row["estimate_pp"], row["bonferroni_ci_low_pp"], row["bonferroni_ci_high_pp"]))
    print("Written: " + str(destination) + " (no model calls)")


if __name__ == "__main__":
    main()
