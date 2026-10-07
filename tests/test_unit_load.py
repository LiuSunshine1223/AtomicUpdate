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
    ACTION_REWRITE_UNIT,
    execute_action,
)
from atomic_memory.unit_load import (  # noqa: E402
    UNIT_LOADS,
    UNIT_LOAD_CONDITIONS,
    build_unit_load_user_prompt,
    condition_matrix,
    execute_and_evaluate_unit_load,
    flattened_fact_order,
    generate_unit_load_case,
    initial_memory_for_unit_load_condition,
    oracle_correct_action,
    oracle_wrong_existing_id_action,
    target_unit_id_for_condition,
    unit_load_action_schema,
    unit_load_condition_spec,
    unit_load_locator_success,
    unit_load_representation_key,
)


class RepresentationEditScopeUnitLoadTests(unittest.TestCase):

    def setUp(self):
        self.case = generate_unit_load_case(
            case_index=0,
            master_seed=20261004,
            targets_per_case=2,
        )
        self.update = self.case["updates"][0]

    def test_01_generation_is_deterministic(self):
        other = generate_unit_load_case(
            case_index=0,
            master_seed=20261004,
            targets_per_case=2,
        )
        self.assertEqual(self.case, other)

    def test_02_three_groupings_have_expected_shapes(self):
        expected_unit_counts = {1: 6, 2: 3, 3: 2}

        for load in UNIT_LOADS:
            representation = self.case["representations"][
                unit_load_representation_key(load)
            ]
            self.assertEqual(
                len(representation),
                expected_unit_counts[load],
            )
            self.assertTrue(
                all(len(unit["fact_ids"]) == load for unit in representation)
            )

    def test_03_shared_fact_order_is_identical_for_all_loads(self):
        shared = list(self.case["shared_fact_order"])
        for load in UNIT_LOADS:
            self.assertEqual(
                flattened_fact_order(self.case, load),
                shared,
            )

    def test_04_same_fact_sentences_across_all_loads(self):
        sentence_sets = []
        for load in UNIT_LOADS:
            representation = self.case["representations"][
                unit_load_representation_key(load)
            ]
            sentences = []
            for unit in representation:
                for sentence in unit["text"].split("."):
                    sentence = sentence.strip()
                    if sentence:
                        sentences.append(sentence + ".")
            sentence_sets.append(sorted(sentences))

        self.assertEqual(sentence_sets[0], sentence_sets[1])
        self.assertEqual(sentence_sets[1], sentence_sets[2])
        self.assertEqual(len(sentence_sets[0]), 6)

    def test_05_representation_freeze_precedes_target_sampling_contract(self):
        invariants = self.case["design_invariants"]
        self.assertTrue(invariants["representations_built_before_target"])
        self.assertTrue(invariants["targets_share_same_initial_memory"])
        self.assertTrue(invariants["shared_fact_order_across_representations"])
        self.assertEqual(invariants["fact_count"], 6)
        self.assertEqual(invariants["unit_loads"], [1, 2, 3])

    def test_06_condition_matrix_is_exactly_six_cells(self):
        matrix = condition_matrix()
        self.assertEqual(len(matrix), 6)
        self.assertEqual(
            [(row[0], row[1]) for row in matrix],
            [
                (1, "select"),
                (1, "patch"),
                (2, "select"),
                (2, "patch"),
                (3, "select"),
                (3, "patch"),
            ],
        )

    def test_07_every_condition_has_one_target_unit(self):
        for condition in UNIT_LOAD_CONDITIONS:
            target_id = target_unit_id_for_condition(
                self.case,
                self.update,
                condition,
            )
            memory = initial_memory_for_unit_load_condition(
                self.case,
                condition,
            )
            self.assertEqual(
                sum(
                    1
                    for unit in memory
                    if unit["memory_id"] == target_id
                ),
                1,
            )

    def test_08_correct_actions_pass_all_six_conditions(self):
        for condition in UNIT_LOAD_CONDITIONS:
            with self.subTest(condition=condition):
                action = oracle_correct_action(
                    self.case,
                    self.update,
                    condition,
                )
                result = execute_and_evaluate_unit_load(
                    self.case,
                    self.update,
                    condition,
                    action,
                )
                self.assertTrue(result["execution"]["legal_action"])
                self.assertTrue(result["execution"]["execution_success"])
                self.assertTrue(result["evaluation"]["structure_valid"])
                self.assertTrue(result["evaluation"]["target_new_current"])
                self.assertTrue(result["evaluation"]["target_old_not_current"])
                self.assertTrue(
                    result["evaluation"]["protected_facts_preserved"]
                )
                self.assertTrue(
                    result["evaluation"]["no_unsupported_changes"]
                )
                self.assertTrue(result["evaluation"]["end_to_end_success"])
                self.assertTrue(
                    unit_load_locator_success(
                        self.case,
                        self.update,
                        condition,
                        action,
                    )
                )

    def test_09_wrong_existing_id_executes_then_fails_all_six_conditions(self):
        for condition in UNIT_LOAD_CONDITIONS:
            with self.subTest(condition=condition):
                action = oracle_wrong_existing_id_action(
                    self.case,
                    self.update,
                    condition,
                )
                result = execute_and_evaluate_unit_load(
                    self.case,
                    self.update,
                    condition,
                    action,
                )

                # Gold target knowledge must not censor execution of a valid
                # action that names an existing but wrong unit.
                self.assertTrue(result["execution"]["legal_action"])
                self.assertTrue(result["execution"]["execution_success"])
                self.assertFalse(
                    unit_load_locator_success(
                        self.case,
                        self.update,
                        condition,
                        action,
                    )
                )
                self.assertFalse(result["evaluation"]["target_new_current"])
                self.assertFalse(result["evaluation"]["target_old_not_current"])
                self.assertFalse(result["evaluation"]["end_to_end_success"])

    def test_10_nonexistent_id_is_illegal_for_both_interfaces(self):
        for action_type, action in (
            (
                ACTION_REWRITE_UNIT,
                {
                    "action": ACTION_REWRITE_UNIT,
                    "memory_id": "not-a-real-id",
                    "text": "Anything",
                },
            ),
            (
                ACTION_EXACT_PATCH,
                {
                    "action": ACTION_EXACT_PATCH,
                    "memory_id": "not-a-real-id",
                    "old_span": "Anything",
                    "new_span": "Replacement",
                },
            ),
        ):
            memory = initial_memory_for_unit_load_condition(
                self.case,
                "L2_select" if action_type == ACTION_REWRITE_UNIT else "L2_patch",
            )
            before = copy.deepcopy(memory)
            result = execute_action(
                initial_memory=memory,
                action=action,
                expected_action_type=action_type,
            )
            self.assertFalse(result["legal_action"])
            self.assertFalse(result["execution_success"])
            self.assertEqual(result["final_memory"], before)

    def test_11_patch_old_target_span_is_unique_in_correct_unit(self):
        for condition in UNIT_LOAD_CONDITIONS:
            spec = unit_load_condition_spec(condition)
            if spec["expected_action_type"] != ACTION_EXACT_PATCH:
                continue

            memory = initial_memory_for_unit_load_condition(
                self.case,
                condition,
            )
            target_id = target_unit_id_for_condition(
                self.case,
                self.update,
                condition,
            )
            target_unit = next(
                unit
                for unit in memory
                if unit["memory_id"] == target_id
            )
            old_text = self.update["old_fact"]["current_text"]
            self.assertEqual(target_unit["text"].count(old_text), 1)

    def test_12_schema_binds_existing_ids_for_all_six_conditions(self):
        for condition in UNIT_LOAD_CONDITIONS:
            with self.subTest(condition=condition):
                memory = initial_memory_for_unit_load_condition(
                    self.case,
                    condition,
                )
                schema = unit_load_action_schema(condition, memory)
                enum_ids = schema["properties"]["memory_id"]["enum"]
                expected_ids = [unit["memory_id"] for unit in memory]
                self.assertEqual(enum_ids, expected_ids)

    def test_13_prompt_binding_hides_latent_metadata(self):
        hidden_tokens = (
            "fact_ids",
            "target_fact_id",
            "expected_current_facts",
            "allowed_historical_facts",
            "case_sha256",
            "representation_hash",
        )

        for condition in UNIT_LOAD_CONDITIONS:
            memory = initial_memory_for_unit_load_condition(
                self.case,
                condition,
            )
            prompt = build_unit_load_user_prompt(
                condition,
                memory,
                self.update["evidence_text"],
            )
            lowered = prompt.lower()
            for hidden in hidden_tokens:
                self.assertNotIn(hidden.lower(), lowered)
            self.assertNotIn(self.update["target_fact_id"], prompt)

    def test_14_two_targets_are_distinct_and_share_initial_representations(self):
        self.assertEqual(len(self.case["updates"]), 2)
        self.assertNotEqual(
            self.case["updates"][0]["target_fact_id"],
            self.case["updates"][1]["target_fact_id"],
        )

        for condition in UNIT_LOAD_CONDITIONS:
            first = initial_memory_for_unit_load_condition(
                self.case,
                condition,
            )
            second = initial_memory_for_unit_load_condition(
                self.case,
                condition,
            )
            self.assertEqual(first, second)
            self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
