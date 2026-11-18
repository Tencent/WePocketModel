"""CPU GGUF language execution with the trained Transformers vision/MGVB path.

Language tensors stay in GGUF and execute in the pinned llama.cpp backend.
Only requested token embedding rows and the BF16 visual/task modules are
decoded to FP32 for the common image, E-slot and J-slot implementations.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import platform
import time
from pathlib import Path

try:
    import gguf
except ModuleNotFoundError as exc:
    if exc.name != "gguf":
        raise
    raise RuntimeError(
        "Install GGUF dependencies: python -m pip install -r inference/requirements-gguf.txt"
    ) from exc
import numpy as np
import torch
from checkpoint import LLAMA_CPP_COMMIT, select_runtime_sources
from common import load_contracts, validate_architecture
from runtime import WePMRuntime
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


class TensorStore:
    """Read GGUF tensors, retaining mmap storage for the language table."""

    def __init__(self, path):
        self.reader = gguf.GGUFReader(str(path))
        self.tensors = {tensor.name: tensor for tensor in self.reader.tensors}
        field = self.reader.fields.get("unimodel.tensor_shapes")
        self.shapes = json.loads(field.contents()) if field else {}

    def tensor(self, name):
        tensor = self.tensors[name]
        shape = self.shapes.get(name, tuple(reversed(tensor.shape)))
        data = gguf.dequantize(tensor.data, tensor.tensor_type)
        return torch.from_numpy(
            np.array(data, dtype=np.float32, copy=True).reshape(shape)
        )

    def embedding(self, ids):
        tensor = self.tensors["token_embd.weight"]
        indices = ids.cpu().numpy()
        unique, inverse = np.unique(indices, return_inverse=True)
        if unique.min() < 0 or unique.max() >= tensor.shape[1]:
            raise ValueError("token index outside GGUF vocabulary")
        # Slice packed rows first: never materialize the whole language model.
        rows = gguf.dequantize(tensor.data[unique], tensor.tensor_type)
        values = rows[inverse].reshape(*indices.shape, int(tensor.shape[0]))
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True))


class NativeText:
    """Own one CPU context through a small, versioned native C ABI."""

    def __init__(self, model, library, context_size, threads):
        self.handle = None
        self.api = ctypes.CDLL(str(library))
        api = self.api
        api.wepm_error.restype = ctypes.c_char_p
        api.wepm_abi_version.restype = ctypes.c_int
        if api.wepm_abi_version() != 1:
            raise ValueError("unsupported WePM GGUF library ABI")
        api.wepm_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        api.wepm_load.restype = ctypes.c_void_p
        api.wepm_free.argtypes = [ctypes.c_void_p]
        for name in ("wepm_width", "wepm_vocabulary"):
            getattr(api, name).argtypes = [ctypes.c_void_p]
            getattr(api, name).restype = ctypes.c_int
        api.wepm_eval.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
        ]
        api.wepm_eval.restype = ctypes.c_int
        self.handle = api.wepm_load(os.fsencode(model), context_size, threads)
        if not self.handle:
            raise RuntimeError(api.wepm_error().decode())
        self.width = api.wepm_width(self.handle)
        self.vocabulary = api.wepm_vocabulary(self.handle)
        self.context_size = context_size

    def evaluate(self, embeddings, positions, *, reset, hidden):
        if not self.handle:
            raise RuntimeError("GGUF context is closed")
        count = embeddings.shape[1]
        if tuple(embeddings.shape) != (1, count, self.width) or tuple(
            positions.shape
        ) != (4, 1, count):
            raise ValueError(
                "GGUF supports one unpadded request with four position axes"
            )
        data = np.ascontiguousarray(embeddings.float().cpu().numpy())
        # Common runtime: sequence, temporal, height, width.
        # Pinned GGUF ZYXT layout: sequence, height, width, temporal.
        pos = np.ascontiguousarray(
            positions[[0, 2, 3, 1], 0].cpu().numpy(), dtype=np.int32
        )
        output = np.empty(self.width if hidden else self.vocabulary, dtype=np.float32)
        code = self.api.wepm_eval(
            self.handle,
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            count,
            int(reset),
            int(hidden),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if code:
            raise RuntimeError(self.api.wepm_error().decode())
        if not np.isfinite(output).all():
            raise ValueError("nonfinite GGUF output")
        return torch.from_numpy(output)

    def close(self):
        if self.handle:
            self.api.wepm_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def library_path(commit, supplied=None):
    if supplied or os.environ.get("WEPM_GGUF_LIBRARY"):
        path = Path(supplied or os.environ["WEPM_GGUF_LIBRARY"]).expanduser().resolve()
    else:
        filename = (
            "libwepm_gguf.dylib" if platform.system() == "Darwin" else "libwepm_gguf.so"
        )
        path = Path.home() / ".cache/wepm-gguf" / commit / "native-build/bin" / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"GGUF CPU library missing: {path}. Run python inference/build_gguf.py, or pass --gguf-library."
        )
    return path


class UniModelGGUF(WePMRuntime):
    """CPU inference with native GGUF language weights and shared task logic."""

    def __init__(
        self,
        model_dir,
        *,
        variant="q4",
        library=None,
        context_size=4096,
        threads=8,
        trust_model_code=False,
    ):
        self.native = None
        self.model_dir = Path(model_dir).resolve()
        manifest = json.loads((self.model_dir / "bundle.json").read_text())
        if (
            manifest.get("format") != "wepm-gguf-release-v1"
            or manifest.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
        ):
            raise ValueError("unsupported WePM GGUF bundle or llama.cpp revision")
        language_name = manifest["language_variants"].get(variant)
        if not language_name:
            raise ValueError(f"GGUF variant {variant!r} is missing from bundle.json")
        auxiliary_name = "multimodal-BF16.gguf"
        for name in (language_name, auxiliary_name):
            path = self.model_dir / name
            expected = manifest["files"].get(name)
            if (
                Path(name).name != name
                or not expected
                or not path.is_file()
                or path.stat().st_size != expected["bytes"]
            ):
                raise ValueError(f"missing or incomplete GGUF weight file: {path}")

        # Check the small assets and code before using them. Do not read several
        # GiB solely to recompute weight checksums at each invocation.
        for directory, table in (
            (self.model_dir, manifest["asset_sha256"]),
            (self.model_dir / "runtime_sources", manifest["runtime_source_sha256"]),
        ):
            for name, expected in table.items():
                if (
                    Path(name).name != name
                    or hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ValueError(f"GGUF asset integrity mismatch: {name}")
        native_library = library_path(manifest["llama_cpp_commit"], library)
        self.config = json.loads((self.model_dir / "config.json").read_text())
        runtime_sources = select_runtime_sources(
            self.model_dir, trust_model_code=trust_model_code
        )
        self.contracts = load_contracts(runtime_sources)
        self.decision_config = validate_architecture(self.config, self.contracts)
        self.device, self.dtype = torch.device("cpu"), torch.float32
        language = self.model_dir / language_name
        self.language = TensorStore(language)
        aux = TensorStore(self.model_dir / auxiliary_name)
        state = {name: aux.tensor(name) for name in aux.tensors}
        vision_config = AutoConfig.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        ).vision_config
        vision_config._attn_implementation = "sdpa"
        self.vision = Qwen3_5VisionModel(vision_config).float().eval()
        prefix = "model.visual."
        visual = {
            name.removeprefix(prefix): state.pop(name)
            for name in list(state)
            if name.startswith(prefix)
        }
        self.vision.load_state_dict(visual, strict=True)
        del visual
        self._load_task_modules(runtime_sources, state)
        self.native = NativeText(language, native_library, context_size, threads)
        text_config = self.config["text_config"]
        if (
            self.native.width != text_config["hidden_size"]
            or self.native.vocabulary != text_config["vocab_size"]
        ):
            self.close()
            raise ValueError("GGUF language dimensions differ from model config")

    def _token_embeddings(self, ids):
        return self.language.embedding(ids)

    def _last_hidden(self, embeddings, positions):
        return self.native.evaluate(embeddings, positions, reset=True, hidden=True)[
            None, :
        ]

    def _vision_features(self, pixels, grid):
        return self.vision(pixels, grid_thw=grid, return_dict=True).pooler_output

    @torch.inference_mode()
    def generate(self, content, max_new_tokens=64):
        if not 1 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be in [1, 512]")
        start = time.monotonic()
        ids, embeddings, positions, metadata = self._prepare_inputs(
            content, task="generate"
        )
        if ids.shape[1] + max_new_tokens > self.native.context_size:
            raise ValueError("prompt and generation exceed --context-size")
        logits = self.native.evaluate(embeddings, positions, reset=True, hidden=False)
        rope_start = int(positions[1:].max()) + 1
        stop_ids = self._stop_token_ids()
        generated, finish = [], "length"
        for index in range(max_new_tokens):
            token = int(logits.argmax())
            generated.append(token)
            if token in stop_ids:
                finish = "eos"
                break
            if index + 1 == max_new_tokens:
                break
            pos = torch.full((4, 1, 1), rope_start + index, dtype=torch.long)
            pos[0] = ids.shape[1] + index
            logits = self.native.evaluate(
                self._token_embeddings(torch.tensor([[token]], device=self.device)),
                pos,
                reset=False,
                hidden=False,
            )
        return {
            "text": self.processor.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
            "token_ids": generated,
            "finish_reason": finish,
            "seconds": time.monotonic() - start,
            "input": metadata,
        }

    def close(self):
        if self.native is not None:
            self.native.close()
