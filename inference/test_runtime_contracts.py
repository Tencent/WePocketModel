"""Regression tests for the inference entry points and shared runtime."""

import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from checkpoint import BUNDLED_RUNTIME, RUNTIME_FILES, select_runtime_sources
from common import write_json
from runtime import UniModelTransformers


class OutputTests(unittest.TestCase):
    def test_concurrent_writers_publish_one_complete_result(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            start = Barrier(4)

            def write(index):
                start.wait()
                try:
                    write_json(output, {"writer": index})
                except FileExistsError:
                    return None
                return index

            with ThreadPoolExecutor(max_workers=4) as pool:
                winners = [i for i in pool.map(write, range(4)) if i is not None]
            self.assertEqual(winners, [json.loads(output.read_text())["writer"]])
            self.assertEqual(list(output.parent.iterdir()), [output])

    def test_serialization_failure_leaves_destination_available(self):
        with TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with self.assertRaises(TypeError):
                write_json(output, {"invalid": object()})
            self.assertEqual(list(output.parent.iterdir()), [])
            write_json(output, {"retry": True})
            self.assertTrue(json.loads(output.read_text())["retry"])


class RuntimeSourceTests(unittest.TestCase):
    def test_external_sources_require_opt_in_and_a_complete_directory(self):
        with TemporaryDirectory() as directory:
            model = Path(directory)
            external = model / "runtime_sources"
            external.mkdir()
            with self.assertRaisesRegex(ValueError, "incomplete WePM runtime"):
                select_runtime_sources(model, trust_model_code=True)
            for name in RUNTIME_FILES:
                (external / name).touch()
            self.assertEqual(
                select_runtime_sources(model, trust_model_code=False), BUNDLED_RUNTIME
            )
            self.assertEqual(
                select_runtime_sources(model, trust_model_code=True), external
            )


class GenerationTests(unittest.TestCase):
    def test_generation_stops_on_config_or_tokenizer_eos(self):
        for eos, tokenizer_eos, finish in (
            (2, None, "eos"),
            ([1, 2], 3, "eos"),
            ([1, 3], 2, "eos"),
            (None, 2, "eos"),
            ([1, 3], None, "length"),
        ):
            with self.subTest(eos=eos, tokenizer_eos=tokenizer_eos):
                runtime = object.__new__(UniModelTransformers)
                runtime.config = {"eos_token_id": eos}
                runtime.processor = Mock()
                runtime.processor.tokenizer.eos_token_id = tokenizer_eos
                runtime._prepare_inputs = Mock(
                    return_value=(
                        torch.zeros(1, 1),
                        torch.zeros(1, 1, 3),
                        torch.zeros(4, 1, 1),
                        {},
                    )
                )
                runtime.model = Mock()
                runtime.model.model.language_model.return_value = SimpleNamespace(
                    past_key_values=None,
                    last_hidden_state=torch.tensor([[[0.0, 0.0, 1.0]]]),
                )
                runtime.model.lm_head = torch.nn.Identity()
                result = runtime.generate([], max_new_tokens=1)
                self.assertEqual(result["token_ids"], [2])
                self.assertEqual(result["finish_reason"], finish)


if __name__ == "__main__":
    unittest.main()
