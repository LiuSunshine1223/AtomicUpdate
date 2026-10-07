import copy
import unittest
from atomic_memory.reanalysis import compare_json

from scripts.exp4_profile import analyze_profile_audit as explanatory


class ProfileExplanatoryTest(unittest.TestCase):
    def test_grammar_failure_preserves_missing_semantic_fields(self):
        evaluation = {
            "contract_valid": False, "target_new_current": None,
            "protected_facts_preserved": None, "target_old_not_current": None,
        }
        original = copy.deepcopy(evaluation)
        self.assertEqual(explanatory.classify_disagreement(evaluation, "Emma"),
                         "grammar_unscorable")
        self.assertEqual(evaluation, original)
        self.assertIsNone(evaluation["target_new_current"])

    def test_contradiction_takes_priority_over_overlapping_fact_loss(self):
        evaluation = {
            "contract_valid": True, "target_old_not_current": False,
            "target_new_current": True, "protected_facts_preserved": False,
            "missing_protected_facts": [{"subject": "Emma"}],
        }
        self.assertEqual(explanatory.classify_disagreement(evaluation, "Emma"),
                         "old_and_new_current")

    def test_other_subject_category_rejects_queried_subject_loss(self):
        evaluation = {
            "contract_valid": True, "target_old_not_current": True,
            "target_new_current": True, "protected_facts_preserved": False,
            "missing_protected_facts": [{"subject": "Emma"}],
        }
        with self.assertRaisesRegex(ValueError, "other subject"):
            explanatory.classify_disagreement(evaluation, "Emma")

    def test_raw_records_reproduce_published_counts_and_paired_intervals(self):
        source = explanatory.PROJECT_ROOT / "outputs/exp4_profile"
        audit, _ = explanatory.build_outputs(source)
        self.assertEqual(audit["counts"], {
            "grammar_unscorable": 26, "old_and_new_current": 14,
            "outside_subject_loss": 11, "unsupported_target_hit": 1,
            "duplicate_fact": 1,
        })
        self.assertEqual(len(audit["rows"]), 53)
        self.assertEqual((audit["cases"], audit["unique_reader_inputs"]), (36, 51))
        # Values printed in the paper, in Llama/Qwen/Gemma, Patch/Select order.
        expected = [(-1, -3, 0), (-14, -21, -8), (0, 0, 0),
                    (-11, -17, -6), (-14, -21, -8), (-3, -7, 0)]
        self.assertEqual(len(audit["exploratory_endpoint_changes"]), len(expected))
        for result, values in zip(audit["exploratory_endpoint_changes"], expected):
            for key, value in zip(("estimate_pp", "ci_low_pp", "ci_high_pp"), values):
                self.assertAlmostEqual(result[key], value)
        for row in audit["rows"]:
            if row["category"] == "grammar_unscorable":
                self.assertIsNone(row["original_evaluation"]["target_new_current"])

    def test_bootstrap_rejects_incomplete_target_pairing(self):
        source = explanatory.PROJECT_ROOT / "outputs/exp4_profile"
        _, _, readers, cases, _ = explanatory.verify_inputs(source)
        incomplete = readers[1:]
        with self.assertRaisesRegex(ValueError, "two targets"):
            explanatory.endpoint_changes(incomplete, cases)



class ReanalysisIntegrityTest(unittest.TestCase):
    def test_producer_location_changes_keep_scientific_comparison(self):
        saved={"contrast": 11.0, "analysis_script_sha256": "old", "outputs": {"analysis": "old/analysis.json"}}
        current={"contrast": 11.0, "analysis_script_sha256": "new", "outputs": {"analysis": "new/analysis.json"}}
        compare_json(current, saved, "analysis")

    def test_changed_estimate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scientific output differs"):
            compare_json({"contrast": 12.0}, {"contrast": 11.0}, "analysis")

    def test_changed_source_hash_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "scientific output differs"):
            compare_json({"source_hashes": {"results": "new"}}, {"source_hashes": {"results": "old"}}, "analysis")


if __name__ == "__main__":
    unittest.main()
