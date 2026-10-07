import copy
import unittest
from unittest import mock

from scripts.exp1_baseline import run_baseline as baseline


class RuntimeCompatibilityTest(unittest.TestCase):
    def setUp(self):
        self.historical = {"model_snapshots": {
            model: {
                "provider": "ollama", "ollama_version": "0.34.0", "model": model,
                "model_digest": "weights-for-" + model, "parameter_size": "8.0B",
                "quantization_level": "Q4_K_M",
            }
            for model in baseline.MODELS
        }}
        self.current = copy.deepcopy(self.historical["model_snapshots"])
        for snapshot in self.current.values():
            snapshot["ollama_version"] = "0.34.4"
        self.backend = mock.Mock()
        self.backend.describe.side_effect = lambda model: dict(self.current[model])

    def test_new_batch_accepts_version_change_and_records_both_versions(self):
        observed = baseline.preflight_models(self.backend, self.historical)
        reference = baseline.integration_runtime_reference(self.historical, observed)
        self.assertEqual(reference["version_changed_models"], list(baseline.MODELS))
        self.assertEqual(set(reference["historical_integration_versions"].values()), {"0.34.0"})
        self.assertEqual(set(reference["current_versions"].values()), {"0.34.4"})
        self.assertEqual(observed, self.current)
        self.assertEqual(set(snapshot["ollama_version"] for snapshot in
                             self.historical["model_snapshots"].values()), {"0.34.0"})

    def test_same_version_still_passes(self):
        self.current = copy.deepcopy(self.historical["model_snapshots"])
        observed = baseline.preflight_models(self.backend, self.historical)
        reference = baseline.integration_runtime_reference(self.historical, observed)
        self.assertEqual(reference["version_changed_models"], [])

    def test_changed_model_identity_is_rejected_despite_compatible_runtime(self):
        model = baseline.MODELS[0]
        for field in ("provider", "model", "model_digest", "parameter_size"):
            with self.subTest(field=field):
                original = self.current[model][field]
                self.current[model][field] = "changed"
                with self.assertRaisesRegex(RuntimeError, field):
                    baseline.preflight_models(self.backend, self.historical)
                self.current[model][field] = original

    def test_changed_quantization_is_rejected(self):
        self.current[baseline.MODELS[0]]["quantization_level"] = "Q4_0"
        with self.assertRaisesRegex(RuntimeError, "Unexpected quantization"):
            baseline.preflight_models(self.backend, self.historical)

    def test_mixed_or_unknown_runtime_is_rejected(self):
        self.current[baseline.MODELS[0]]["ollama_version"] = "0.34.5"
        with self.assertRaisesRegex(RuntimeError, "same known Ollama version"):
            baseline.preflight_models(self.backend, self.historical)
        for snapshot in self.current.values():
            snapshot["ollama_version"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "same known Ollama version"):
            baseline.preflight_models(self.backend, self.historical)

    def test_full_snapshot_retains_runtime_for_freeze_and_result_records(self):
        model = baseline.MODELS[0]
        frozen = baseline.snapshot_fingerprint(self.current[model])
        historical = baseline.snapshot_fingerprint(self.historical["model_snapshots"][model])
        self.assertEqual(frozen["ollama_version"], "0.34.4")
        self.assertNotEqual(frozen, historical)
        self.assertEqual(baseline.model_identity_fingerprint(self.current[model]),
                         baseline.model_identity_fingerprint(self.historical["model_snapshots"][model]))

    def test_resume_rejects_version_change_after_current_batch_freeze(self):
        frozen = {"model_snapshots": {
            model: baseline.snapshot_fingerprint(snapshot)
            for model, snapshot in self.current.items()
        }}
        manifest_path = mock.Mock()
        manifest_path.exists.return_value = True
        with mock.patch.object(baseline, "FREEZE_MANIFEST_PATH", manifest_path), \
                mock.patch.object(baseline, "read_json", return_value={"immutable_protocol": frozen}):
            self.assertEqual(baseline.freeze_or_verify(frozen), "VERIFIED_EXISTING")
            updated = copy.deepcopy(frozen)
            for snapshot in updated["model_snapshots"].values():
                snapshot["ollama_version"] = "0.34.5"
            with self.assertRaisesRegex(RuntimeError, "changed after freeze"):
                baseline.freeze_or_verify(updated)


if __name__ == "__main__":
    unittest.main()
