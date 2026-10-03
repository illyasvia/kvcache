import json
import tempfile
import unittest
from pathlib import Path

from evaluation.datasets import (
    BenchmarkSample,
    get_adapter,
    resolve_dataset_source,
    supported_datasets,
)
from evaluation.runner import predict_sample, run_benchmark
from evaluation.scoring import anls_score, multiple_choice_score, vqa_score
from multimodal_eval import parse_args


class FakeMultimodalModel:
    def __init__(self, answers=None):
        self.model_path = "fake-model"
        self.answers = iter(answers or ["invoice number"])
        self.single_calls = 0
        self.multiple_calls = 0
        self.last_input_tokens = 10
        self.last_output_tokens = 2

    def process_multimodel(self, prompt, image, max_new_tokens):
        self.single_calls += 1
        return [next(self.answers)]

    def process_multiple_multimodel(self, prompt, images, max_new_tokens):
        self.multiple_calls += 1
        return [next(self.answers)]


class DatasetAdapterTests(unittest.TestCase):
    def test_registry_covers_supported_multimodal_benchmarks(self):
        self.assertEqual(
            supported_datasets(),
            ("blink", "docvqa", "mathvista", "mmmu", "mmstar", "textvqa"),
        )

    def test_docvqa_adapter_builds_single_image_sample(self):
        adapter = get_adapter("docvqa")
        sample = adapter.to_sample({
            "questionId": 42,
            "question": "What is the invoice number?",
            "image": "image-object",
            "answers": ["A-100", "A100"],
        }, 3)

        self.assertEqual(sample.sample_id, "3:42")
        self.assertEqual(sample.images, ("image-object",))
        self.assertEqual(sample.answers, ("A-100", "A100"))
        self.assertIn("document image", sample.prompt)

    def test_mmmu_adapter_collects_images_and_renders_options(self):
        adapter = get_adapter("mmmu")
        sample = adapter.to_sample({
            "id": "sample-1",
            "question": "Which answer is correct?",
            "options": "['first', 'second']",
            "image_1": "first-image",
            "image_2": None,
            "image_3": "third-image",
            "answer": "B",
        }, 0)

        self.assertEqual(sample.images, ("first-image", "third-image"))
        self.assertIn("(A) first", sample.prompt)
        self.assertIn("(B) second", sample.prompt)
        self.assertEqual(sample.answers, ("B",))

    def test_dataset_root_resolves_local_dataset_path(self):
        adapter = get_adapter("docvqa")
        self.assertEqual(
            resolve_dataset_source(adapter, dataset_root="/datasets"),
            "/datasets/lmms-lab/DocVQA",
        )


class ScoringTests(unittest.TestCase):
    def test_docvqa_anls_uses_best_reference_and_threshold(self):
        self.assertEqual(anls_score("invoice 123", ["invoice 124", "other"]), 90.9090909090909)
        self.assertEqual(anls_score("A-100", ["A100"]), 80.0)
        self.assertEqual(anls_score("unrelated", ["invoice 124"]), 0.0)

    def test_textvqa_consensus_score_caps_at_one(self):
        self.assertAlmostEqual(vqa_score("cat", ["cat", "Cat", "dog"]), 200 / 3)
        self.assertEqual(vqa_score("cat", ["cat", "cat", "cat", "cat"]), 100.0)

    def test_multiple_choice_extracts_answer_letter(self):
        self.assertEqual(multiple_choice_score("The answer is (B).", ["B"]), 100.0)
        self.assertEqual(multiple_choice_score("Option C", ["B"]), 0.0)


class RunnerTests(unittest.TestCase):
    def test_predict_sample_routes_multiple_images(self):
        model = FakeMultimodalModel(["B"])
        sample = BenchmarkSample("0", "question", ("a", "b"), ("B",))

        prediction = predict_sample(model, sample, max_new_tokens=8)

        self.assertEqual(prediction, "B")
        self.assertEqual(model.multiple_calls, 1)
        self.assertEqual(model.single_calls, 0)

    def test_run_benchmark_checkpoints_scores_and_resumes(self):
        records = [{
            "questionId": 1,
            "question": "What is shown?",
            "image": "image",
            "answers": ["invoice number"],
        }]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.json"
            first_model = FakeMultimodalModel(["invoice number"])

            result = run_benchmark(
                first_model,
                get_adapter("docvqa"),
                records,
                output_path,
                limit=1,
                max_new_tokens=8,
            )

            self.assertEqual(first_model.single_calls, 1)
            self.assertEqual(result["_summary"]["average_score"], 100.0)
            saved = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["samples"]["0:1"]["input_tokens"], 10)

            resumed_model = FakeMultimodalModel([])
            resumed = run_benchmark(
                resumed_model,
                get_adapter("docvqa"),
                records,
                output_path,
                limit=1,
                max_new_tokens=8,
                resume=True,
            )

            self.assertEqual(resumed_model.single_calls, 0)
            self.assertEqual(resumed["_summary"]["processed_in_this_run"], 0)


class CliTests(unittest.TestCase):
    def test_parse_args_supports_docvqa_smoke_run(self):
        args = parse_args([
            "--dataset", "docvqa",
            "--model_type", "qwen3vl",
            "--limit", "5",
        ])

        self.assertEqual(args.dataset, "docvqa")
        self.assertEqual(args.limit, 5)
        self.assertEqual(args.max_new_tokens, 128)


if __name__ == "__main__":
    unittest.main()
