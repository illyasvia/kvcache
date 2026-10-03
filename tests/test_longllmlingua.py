import unittest
from types import SimpleNamespace

import torch
from transformers import GPT2Config

from compression.longllmlingua import (
    LongLLMLinguaCompressor,
    LongLLMLinguaConfig,
    _to_dynamic_cache,
    _to_legacy_cache,
)
from compression.needle_prompt import PromptParseError, parse_needle_prompt


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(char) for char in text]}

    def decode(self, token_ids, **kwargs):
        return "".join(chr(token_id) for token_id in token_ids)


class FakeBackend:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.last_kwargs = None
        self.recover_args = None

    def compress_prompt(self, contexts, **kwargs):
        self.last_kwargs = {"contexts": contexts, **kwargs}
        compressed = "COMPRESSED\n\n" + kwargs["question"]
        return {
            "compressed_prompt": compressed,
            "origin_tokens": 100,
            "compressed_tokens": 25,
        }

    def recover(self, original_prompt, compressed_prompt, response):
        self.recover_args = (original_prompt, compressed_prompt, response)
        return f"RECOVERED:{response}"


def make_prompt(context="abcdefghij"):
    return (
        "You are helpful.\n"
        "The document given to you by the user is \n"
        f"{context}\n\n"
        "Now, the question is: What is hidden?"
    )


class NeedlePromptParserTests(unittest.TestCase):
    def test_parse_prompt(self):
        parsed = parse_needle_prompt(make_prompt("needle and haystack"))

        self.assertTrue(parsed.instruction.endswith("The document given to you by the user is"))
        self.assertEqual(parsed.context, "needle and haystack")
        self.assertEqual(parsed.question, "Now, the question is: What is hidden?")

    def test_parse_chinese_prompt(self):
        prompt = (
            "你是一个智能助手\n用户现在给你的文档是\n"
            "针在干草堆里\n\n现在请问：针在哪里？"
        )
        parsed = parse_needle_prompt(prompt)

        self.assertEqual(parsed.context, "针在干草堆里")
        self.assertEqual(parsed.question, "现在请问：针在哪里？")

    def test_missing_marker_raises(self):
        with self.assertRaises(PromptParseError):
            parse_needle_prompt("plain prompt")


class LongLLMLinguaCompressorTests(unittest.TestCase):
    def test_compression_uses_question_aware_parameters(self):
        backend = FakeBackend()
        config = LongLLMLinguaConfig(
            model_name="fake",
            device="cpu",
            rate=0.25,
            chunk_tokens=4,
            iterative_size=2,
            reorder_context="sort",
            dynamic_context_compression_ratio=0.3,
        )
        compressor = LongLLMLinguaCompressor(config, backend=backend)

        result = compressor.compress(make_prompt())

        self.assertEqual(backend.last_kwargs["contexts"], ["abcd", "efgh", "ij"])
        self.assertEqual(backend.last_kwargs["rate"], 0.25)
        self.assertEqual(backend.last_kwargs["condition_in_question"], "after_condition")
        self.assertTrue(backend.last_kwargs["condition_compare"])
        self.assertEqual(backend.last_kwargs["rank_method"], "longllmlingua")
        self.assertEqual(result.compressed_prompt, "COMPRESSED\n\nNow, the question is: What is hidden?")
        self.assertEqual(result.info["context_chunks"], 3)
        self.assertEqual(result.ratio, 4.0)

    def test_recovery_can_be_enabled_or_disabled(self):
        prompt = make_prompt()
        backend = FakeBackend()
        enabled = LongLLMLinguaCompressor(
            LongLLMLinguaConfig(model_name="fake", device="cpu", chunk_tokens=20),
            backend=backend,
        )
        result = enabled.compress(prompt)
        self.assertEqual(enabled.recover(result, "answer"), "RECOVERED:answer")

        disabled = LongLLMLinguaCompressor(
            LongLLMLinguaConfig(
                model_name="fake",
                device="cpu",
                chunk_tokens=20,
                enable_recovery=False,
            ),
            backend=backend,
        )
        self.assertEqual(disabled.recover(result, "answer"), "answer")

    def test_invalid_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            LongLLMLinguaCompressor(
                LongLLMLinguaConfig(model_name="fake", device="cpu", rate=0),
                backend=FakeBackend(),
            )


class TransformersCacheCompatibilityTests(unittest.TestCase):
    def test_legacy_cache_round_trip(self):
        key = torch.zeros(1, 2, 3, 4)
        value = torch.ones(1, 2, 3, 4)
        model = SimpleNamespace(config=GPT2Config(n_layer=1, n_head=2, n_embd=8))

        dynamic_cache = _to_dynamic_cache([[key, value]], model)
        restored = _to_legacy_cache(dynamic_cache)

        self.assertEqual(dynamic_cache.get_seq_length(), 3)
        self.assertTrue(torch.equal(restored[0][0], key))
        self.assertTrue(torch.equal(restored[0][1], value))


if __name__ == "__main__":
    unittest.main()
