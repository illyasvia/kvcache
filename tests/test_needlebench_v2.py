import json
import tempfile
import unittest
from pathlib import Path

from compression.needle_prompt import parse_needle_prompt
from dataset.needle.download_needlebench_v2 import build_prompt, generate_dataset
from dataset.needle.generate_comparison_report import build_report
from main import evaluate, inference, sample_keys


class NeedleBenchV2PromptTests(unittest.TestCase):
    def test_parse_official_english_prompt(self):
        prompt = build_prompt(
            "The needle is inside this document.",
            "Where is the needle?",
            "English",
        )

        parsed = parse_needle_prompt(prompt)

        self.assertEqual(parsed.context, "The needle is inside this document.")
        self.assertEqual(
            parsed.question,
            "</Document>\n\nBased on the information in the document, now please answer: "
            "Where is the needle?",
        )


class NeedleBenchV2GenerationTests(unittest.TestCase):
    def test_generate_dataset_uses_official_shape_and_token_budget(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            haystack = {"text": "haystack " * 200}
            needle = {
                "language": "English",
                "needle": "The pass key is cobalt.",
                "retrieval_question": "What is the pass key?",
                "arg2": "cobalt",
            }
            (source_dir / "PaulGrahamEssays.jsonl").write_text(
                json.dumps(haystack) + "\n",
                encoding="utf-8",
            )
            (source_dir / "needles.jsonl").write_text(
                json.dumps(needle) + "\n",
                encoding="utf-8",
            )

            output_path = generate_dataset(
                source_dir=source_dir,
                output_dir=output_dir,
                context_length=3050,
                depth=50,
                repeats=2,
                language_key="en",
            )
            data = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(sample_keys(data), ["0", "1"])
            self.assertEqual(data["0"]["gold"].rsplit("*", 1)[-1], "cobalt")
            self.assertEqual(data["_dataset_info"]["nominal_context_tokens"], 3050)
            self.assertEqual(data["_dataset_info"]["actual_context_tokens_min"], 50)
            self.assertEqual(data["_dataset_info"]["actual_context_tokens_max"], 50)


class FakeBenchmarkModel:
    def __init__(self):
        self.process_calls = 0
        self.last_input_tokens = 123
        self.last_output_tokens = 4
        self.last_compression_info = None
        self.last_h2o_info = None

    def process(self, prompt, max_new_tokens=None):
        self.process_calls += 1
        return "cobalt"


class BenchmarkCheckpointTests(unittest.TestCase):
    def test_inference_checkpoints_and_resumes_completed_samples(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_path = root / "dataset.json"
            result_path = root / "result.json"
            sample = {
                "origin_prompt": [{"role": "HUMAN", "prompt": "prompt"}],
                "gold": "answer*cobalt",
            }
            dataset_path.write_text(
                json.dumps({"0": sample, "1": sample}),
                encoding="utf-8",
            )
            first_model = FakeBenchmarkModel()

            inference(first_model, dataset_path, result_path, max_new_tokens=8)
            first_result = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(first_model.process_calls, 2)
            self.assertEqual(first_result["_run_stats"]["completed_samples"], 2)
            self.assertEqual(first_result["_run_stats"]["input_tokens_total"], 246)

            resumed_model = FakeBenchmarkModel()
            inference(resumed_model, dataset_path, result_path, max_new_tokens=8, resume=True)

            self.assertEqual(resumed_model.process_calls, 0)


class ComparisonReportTests(unittest.TestCase):
    def test_report_summarizes_partial_mode_results(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset_dir = root / "dataset"
            output_root = root / "output"
            mode_dir = output_root / "needlebench_v2_1000k_h2o"
            dataset_dir.mkdir()
            mode_dir.mkdir(parents=True)
            manifest_path = dataset_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps({"revision": "test-revision"}), encoding="utf-8"
            )
            source_path = dataset_dir / "Length1000000Depth0_origin_en_1000k.json"
            source_path.write_text(
                json.dumps({
                    "_dataset_info": {
                        "actual_context_tokens_min": 446324,
                        "actual_context_tokens_max": 446329,
                    }
                }),
                encoding="utf-8",
            )
            result_path = mode_dir / "Length1000000Depth0.json"
            result_path.write_text(
                json.dumps({
                    "0": {
                        "pre": "cobalt",
                        "gold": "answer*cobalt",
                        "inference_seconds": 12.5,
                        "input_tokens": 453544,
                    },
                    "_memory_stats": {"peak_process_rss_gb": 6.25},
                }),
                encoding="utf-8",
            )

            report = build_report(output_root, manifest_path)

            self.assertIn("test-revision", report)
            self.assertIn("H2O | 1/30 | 100.00%", report)
            self.assertIn("446,324–446,329", report)


class NeedleBenchV2ScoringTests(unittest.TestCase):
    def test_official_scoring_requires_complete_case_sensitive_keyword(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result_path = Path(temporary_directory) / "result.json"
            result_path.write_text(
                json.dumps({
                    "0": {"pre": "The key is cobalt.", "gold": "answer*cobalt"},
                    "1": {"pre": "The key is Cobalt.", "gold": "answer*cobalt"},
                }),
                encoding="utf-8",
            )

            average = evaluate(result_path, scoring="needlebench_v2")
            data = json.loads(result_path.read_text(encoding="utf-8"))

            self.assertEqual(average, 50.0)
            self.assertEqual(data["0"]["score"], 100.0)
            self.assertEqual(data["1"]["score"], 0.0)
            self.assertEqual(data["_eval_stats"]["num_samples"], 2)


if __name__ == "__main__":
    unittest.main()
