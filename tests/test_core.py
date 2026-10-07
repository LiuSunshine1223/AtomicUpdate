#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from atomic_memory.core import (  # noqa: E402
    ACTION_EXACT_PATCH,
    ACTION_REWRITE_ALL,
    ACTION_REWRITE_UNIT,
    CONDITION_A,
    execute_action,
    execute_and_evaluate,
    generate_case,
    initial_memory_for_condition,
    parse_memory,
    target_unit_ids,
)


class RepresentationEditScopeCoreTests(unittest.TestCase):

    def setUp(self):
        self.case = generate_case(
            case_index=0,
            master_seed=20260926,
            targets_per_case=2,
        )
        self.update = self.case["updates"][0]
        self.compound = initial_memory_for_condition(
            self.case,
            CONDITION_A,
        )

    def _correct_global_action(self):
        target_fact_id = self.update["target_fact_id"]
        old_text = self.update["old_fact"]["current_text"]
        new_text = self.update["new_fact"]["current_text"]

        units = []

        for unit in self.compound:
            text = unit["text"]

            if target_fact_id in unit["fact_ids"]:
                text = text.replace(
                    old_text,
                    new_text,
                    1,
                )

            units.append(
                {
                    "memory_id": unit["memory_id"],
                    "text": text,
                }
            )

        return {
            "action": ACTION_REWRITE_ALL,
            "units": units,
        }

    def test_01_generation_is_deterministic(self):
        other = generate_case(
            case_index=0,
            master_seed=20260926,
            targets_per_case=2,
        )

        self.assertEqual(
            self.case,
            other,
        )

    def test_02_representation_shapes(self):
        compound = self.case["representations"]["compound"]
        atomic = self.case["representations"]["atomic"]

        self.assertEqual(len(compound), 3)
        self.assertEqual(len(atomic), 6)

        self.assertTrue(
            all(
                len(unit["fact_ids"]) == 2
                for unit in compound
            )
        )

        self.assertTrue(
            all(
                len(unit["fact_ids"]) == 1
                for unit in atomic
            )
        )

    def test_03_same_fact_sentences_only_grouping_differs(self):
        compound_sentences = []

        for unit in self.case["representations"]["compound"]:
            for sentence in unit["text"].split("."):
                sentence = sentence.strip()
                if sentence:
                    compound_sentences.append(
                        sentence + "."
                    )

        atomic_sentences = [
            unit["text"]
            for unit in self.case["representations"]["atomic"]
        ]

        self.assertEqual(
            sorted(compound_sentences),
            sorted(atomic_sentences),
        )

    def test_04_correct_global_rewrite_passes(self):
        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=self._correct_global_action(),
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertTrue(
            result["evaluation"]["end_to_end_success"]
        )

    def test_05_noop_fails_target_update(self):
        action = {
            "action": ACTION_REWRITE_ALL,
            "units": [
                {
                    "memory_id": unit["memory_id"],
                    "text": unit["text"],
                }
                for unit in self.compound
            ],
        }

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        evaluation = result["evaluation"]

        self.assertFalse(
            evaluation["target_new_current"]
        )
        self.assertFalse(
            evaluation["target_old_not_current"]
        )
        self.assertFalse(
            evaluation["end_to_end_success"]
        )

    def test_06_old_and_new_both_current_fails(self):
        action = self._correct_global_action()
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        for unit in action["units"]:
            if unit["memory_id"] == target_unit:
                unit["text"] += (
                    " "
                    + self.update["old_fact"]["current_text"]
                )

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        evaluation = result["evaluation"]

        self.assertTrue(
            evaluation["target_new_current"]
        )
        self.assertFalse(
            evaluation["target_old_not_current"]
        )
        self.assertFalse(
            evaluation["end_to_end_success"]
        )

    def test_07_old_value_as_history_is_allowed(self):
        action = self._correct_global_action()
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        history_text = (
            self.update["allowed_historical_facts"][0]
            ["historical_text"]
        )

        for unit in action["units"]:
            if unit["memory_id"] == target_unit:
                unit["text"] += " " + history_text

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertTrue(
            result["evaluation"]["end_to_end_success"]
        )

    def test_08_missing_protected_fact_fails(self):
        action = self._correct_global_action()
        target_fact_id = self.update["target_fact_id"]
        removed = False

        for unit in action["units"]:
            original_unit = next(
                x
                for x in self.compound
                if x["memory_id"] == unit["memory_id"]
            )

            for fact_id in original_unit["fact_ids"]:
                if fact_id == target_fact_id:
                    continue

                protected = next(
                    fact
                    for fact in self.case["facts"]
                    if fact["fact_id"] == fact_id
                )

                if protected["current_text"] in unit["text"]:
                    unit["text"] = (
                        unit["text"]
                        .replace(
                            protected["current_text"],
                            "",
                            1,
                        )
                        .strip()
                    )
                    removed = True
                    break

            if removed:
                break

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertFalse(
            result["evaluation"]["protected_facts_preserved"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_09_wrong_subject_fails(self):
        action = self._correct_global_action()
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        correct = self.update["new_fact"]["current_text"]
        wrong = correct.replace(
            self.update["new_fact"]["subject"],
            "WrongPerson",
            1,
        )

        for unit in action["units"]:
            if unit["memory_id"] == target_unit:
                unit["text"] = unit["text"].replace(
                    correct,
                    wrong,
                    1,
                )

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertFalse(
            result["evaluation"]["target_new_current"]
        )
        self.assertFalse(
            result["evaluation"]["no_unsupported_changes"]
        )

    def test_10_unsupported_addition_fails(self):
        action = self._correct_global_action()

        action["units"][0]["text"] += (
            " ExtraPerson's current city is Seattle."
        )

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertFalse(
            result["evaluation"]["no_unsupported_changes"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_11_free_form_output_is_contract_failure(self):
        action = self._correct_global_action()

        action["units"][0]["text"] += (
            " This is free-form commentary."
        )

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertFalse(
            result["evaluation"]["contract_valid"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_12_wrong_selected_unit_executes_but_semantics_fail(self):
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        wrong_unit = next(
            unit
            for unit in self.compound
            if unit["memory_id"] != target_unit
        )

        action = {
            "action": ACTION_REWRITE_UNIT,
            "memory_id": wrong_unit["memory_id"],
            "text": (
                wrong_unit["text"]
                + " "
                + self.update["new_fact"]["current_text"]
            ),
        }

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_UNIT,
            update=self.update,
        )

        self.assertTrue(
            result["execution"]["execution_success"]
        )
        self.assertFalse(
            result["evaluation"]["target_old_not_current"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_13_invalid_memory_id_is_illegal(self):
        action = {
            "action": ACTION_REWRITE_UNIT,
            "memory_id": "not-a-real-id",
            "text": "Anything",
        }

        result = execute_action(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_UNIT,
        )

        self.assertFalse(
            result["legal_action"]
        )
        self.assertFalse(
            result["execution_success"]
        )

    def test_14_patch_absent_span_fails_and_rolls_back(self):
        before = copy.deepcopy(self.compound)

        action = {
            "action": ACTION_EXACT_PATCH,
            "memory_id": self.compound[0]["memory_id"],
            "old_span": "This span does not exist.",
            "new_span": "Replacement.",
        }

        result = execute_action(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_EXACT_PATCH,
        )

        self.assertTrue(
            result["legal_action"]
        )
        self.assertFalse(
            result["execution_success"]
        )
        self.assertEqual(
            result["final_memory"],
            before,
        )

    def test_15_patch_repeated_span_fails_and_rolls_back(self):
        before = copy.deepcopy(self.compound)

        action = {
            "action": ACTION_EXACT_PATCH,
            "memory_id": self.compound[0]["memory_id"],
            "old_span": "current",
            "new_span": "previous",
        }

        result = execute_action(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_EXACT_PATCH,
        )

        self.assertTrue(
            result["legal_action"]
        )
        self.assertFalse(
            result["execution_success"]
        )
        self.assertEqual(
            result["final_memory"],
            before,
        )

    def test_16_patch_executes_but_wrong_semantics_fail(self):
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        action = {
            "action": ACTION_EXACT_PATCH,
            "memory_id": target_unit,
            "old_span": self.update["old_fact"]["current_text"],
            "new_span": (
                self.update["allowed_historical_facts"][0]
                ["historical_text"]
            ),
        }

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_EXACT_PATCH,
            update=self.update,
        )

        self.assertTrue(
            result["execution"]["execution_success"]
        )
        self.assertTrue(
            result["evaluation"]["target_old_not_current"]
        )
        self.assertFalse(
            result["evaluation"]["target_new_current"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_17_parser_rejects_unknown_attribute(self):
        memory = [
            {
                "memory_id": "m0",
                "text": (
                    "Emma's current favorite color is blue."
                ),
            }
        ]

        result = parse_memory(memory)

        self.assertFalse(
            result["contract_valid"]
        )

    def test_18_unit_fact_count_change_fails_structure(self):
        action = self._correct_global_action()

        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        other_unit = next(
            unit
            for unit in action["units"]
            if unit["memory_id"] != target_unit
        )

        moved_sentence = other_unit["text"].split(".")[0].strip() + "."
        other_unit["text"] = (
            other_unit["text"]
            .replace(moved_sentence, "", 1)
            .strip()
        )

        for unit in action["units"]:
            if unit["memory_id"] == target_unit:
                unit["text"] = (
                    unit["text"]
                    + " "
                    + moved_sentence
                )

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertTrue(
            result["evaluation"]["contract_valid"]
        )
        self.assertFalse(
            result["evaluation"]["structure_valid"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_19_duplicate_fact_fails_unsupported_change(self):
        action = self._correct_global_action()

        protected_fact = next(
            fact
            for fact in self.update["expected_current_facts"]
            if fact["fact_id"] != self.update["target_fact_id"]
        )
        duplicate_sentence = protected_fact["current_text"]

        target_unit = next(
            unit
            for unit in action["units"]
            if duplicate_sentence in unit["text"]
        )
        target_unit["text"] += " " + duplicate_sentence

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        self.assertGreater(
            result["evaluation"]["duplicate_fact_count"],
            0,
        )
        self.assertFalse(
            result["evaluation"]["no_unsupported_changes"]
        )
        self.assertFalse(
            result["evaluation"]["end_to_end_success"]
        )

    def test_20_shared_fact_order_is_identical_across_representations(self):
        shared_order = self.case["shared_fact_order"]

        compound_order = []
        for unit in self.case["representations"]["compound"]:
            compound_order.extend(unit["fact_ids"])

        atomic_order = [
            unit["fact_ids"][0]
            for unit in self.case["representations"]["atomic"]
        ]

        self.assertEqual(
            compound_order,
            shared_order,
        )
        self.assertEqual(
            atomic_order,
            shared_order,
        )

    def test_21_same_counts_but_cross_unit_move_fails_membership(self):
        action = self._correct_global_action()

        left = action["units"][0]
        right = action["units"][1]

        left_sentences = [
            part.strip() + "."
            for part in left["text"].split(".")
            if part.strip()
        ]
        right_sentences = [
            part.strip() + "."
            for part in right["text"].split(".")
            if part.strip()
        ]

        left_sentences[0], right_sentences[0] = (
            right_sentences[0],
            left_sentences[0],
        )

        left["text"] = " ".join(left_sentences)
        right["text"] = " ".join(right_sentences)

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        evaluation = result["evaluation"]

        self.assertTrue(
            evaluation["contract_valid"]
        )
        self.assertTrue(
            all(
                count == 2
                for count in evaluation[
                    "per_unit_fact_counts"
                ].values()
            )
        )
        self.assertFalse(
            evaluation["membership_valid"]
        )
        self.assertFalse(
            evaluation["structure_valid"]
        )
        self.assertTrue(
            evaluation["protected_facts_preserved"]
        )
        self.assertTrue(
            evaluation["no_unsupported_changes"]
        )
        self.assertFalse(
            evaluation["end_to_end_success"]
        )

    def test_22_stale_old_is_not_unsupported_change(self):
        action = {
            "action": ACTION_REWRITE_ALL,
            "units": [
                {
                    "memory_id": unit["memory_id"],
                    "text": unit["text"],
                }
                for unit in self.compound
            ],
        }

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_ALL,
            update=self.update,
        )

        evaluation = result["evaluation"]

        self.assertFalse(
            evaluation["target_new_current"]
        )
        self.assertFalse(
            evaluation["target_old_not_current"]
        )
        self.assertTrue(
            evaluation["no_unsupported_changes"]
        )
        self.assertEqual(
            evaluation["unsupported_facts"],
            [],
        )
        self.assertIn(
            "target_new_missing",
            evaluation["failure_reasons"],
        )
        self.assertIn(
            "stale_old_retained",
            evaluation["failure_reasons"],
        )
        self.assertNotIn(
            "unsupported_change",
            evaluation["failure_reasons"],
        )

    def test_23_failure_reasons_can_overlap_without_double_count_semantics(self):
        target_unit = target_unit_ids(
            self.case,
            self.update,
        )["compound"][0]

        wrong_unit = next(
            unit
            for unit in self.compound
            if unit["memory_id"] != target_unit
        )

        protected_fact_id = wrong_unit["fact_ids"][0]
        protected_fact = next(
            fact
            for fact in self.update["expected_current_facts"]
            if fact["fact_id"] == protected_fact_id
        )

        wrong_text = wrong_unit["text"].replace(
            protected_fact["current_text"],
            self.update["new_fact"]["current_text"],
            1,
        )

        action = {
            "action": ACTION_REWRITE_UNIT,
            "memory_id": wrong_unit["memory_id"],
            "text": wrong_text,
        }

        result = execute_and_evaluate(
            initial_memory=self.compound,
            action=action,
            expected_action_type=ACTION_REWRITE_UNIT,
            update=self.update,
        )

        reasons = result["evaluation"]["failure_reasons"]

        self.assertIn(
            "structure_failure",
            reasons,
        )
        self.assertIn(
            "stale_old_retained",
            reasons,
        )
        self.assertIn(
            "protected_fact_loss",
            reasons,
        )
        self.assertNotIn(
            "unsupported_change",
            reasons,
        )
        self.assertEqual(
            len(reasons),
            len(set(reasons)),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
