import tempfile
import unittest
import os
from pathlib import Path
from unittest.mock import patch

from analysis.boltz_affinity import BoltzRecord, BoltzResult, affinity_reward, cache_key, score_batch


class BoltzAffinityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.config = {"cache_dir": self.directory.name, "boltz_commit": "pinned", "prediction": {"seed": 0}}
        self.record = BoltzRecord("target", "MPEPTIDE", "CCO", record_id="example")

    def tearDown(self):
        self.directory.cleanup()

    def test_cache_key_changes_with_inference_parameters(self):
        changed = {**self.config, "prediction": {"seed": 1}}
        self.assertNotEqual(cache_key(self.record, self.config), cache_key(self.record, changed))

    def test_cached_result_does_not_start_subprocess(self):
        key = cache_key(self.record, self.config)
        path = Path(self.directory.name) / "scores" / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"affinity_pred_value": -1.2, "affinity_probability_binary": 0.9}', encoding="utf-8")
        result = score_batch([self.record], Path(self.directory.name) / "work", self.config)[0]
        self.assertTrue(result.cached)
        self.assertEqual(result.optimization_value, 1.2)

    def test_paper_reward_uses_first_head(self):
        result = BoltzResult(self.record, "key", affinity_pred_value1=-2.0, affinity_probability_binary1=0.5)
        self.assertEqual(affinity_reward(result), 0.5)

    @patch("analysis.boltz_affinity.subprocess.run")
    def test_batch_reads_all_raw_boltz_heads(self, run):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        output = Path(self.directory.name) / "work" / "boltz-123-x" / "output"

        def fake_mkdir(path, *args, **kwargs):
            path = Path(path)
            original_mkdir(path, *args, **kwargs)
            if path.name == "inputs":
                affinity = path.parent / "output" / "boltz_results_inputs" / "predictions" / "example"
                os.makedirs(affinity, exist_ok=True)
                (affinity / "affinity_example.json").write_text('{"affinity_pred_value": -1.0, "affinity_probability_binary": 0.8, "affinity_pred_value1": -1.1, "affinity_pred_value2": -0.9}', encoding="utf-8")

        from pathlib import Path as PathClass
        original_mkdir = PathClass.mkdir
        with patch("analysis.boltz_affinity.time.time", return_value=123), patch("analysis.boltz_affinity.uuid.uuid4") as token, patch("analysis.boltz_affinity.Path.mkdir", new=fake_mkdir):
            token.return_value.hex = "xxxxxxxx"
            result = score_batch([self.record], Path(self.directory.name) / "work", self.config)[0]
        self.assertEqual(result.affinity_pred_value2, -0.9)
        self.assertFalse(result.cached)


if __name__ == "__main__":
    unittest.main()
