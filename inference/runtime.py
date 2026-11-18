"""Shared WePM tasks and the Transformers CPU/CUDA backend."""

from __future__ import annotations

import io
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from checkpoint import WePMCheckpoint, select_runtime_sources
from common import load_contracts, load_runtime_sources, validate_architecture
from model_source import resolve_model_path
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

try:
    from pillow_heif import register_heif_opener
except ImportError:
    register_heif_opener = None
else:
    register_heif_opener()


def prepare_content(content, prepare_image):
    """Load images without changing the declared text/image order."""
    processor_content = []
    images = []
    content_types = []
    for item in content:
        item_type = item.get("type") if isinstance(item, dict) else None
        if item_type == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text content must be nonempty")
            processor_content.append({"type": "text", "text": text})
        elif item_type == "image":
            source = item.get("path", item.get("bytes"))
            if source is None:
                raise ValueError("image content requires path or bytes")
            if (
                register_heif_opener is None
                and not isinstance(source, bytes)
                and Path(source).suffix.lower() in {".heic", ".heif"}
            ):
                raise RuntimeError(
                    "HEIC input requires pillow-heif; install requirements.txt"
                )
            with Image.open(
                io.BytesIO(source) if isinstance(source, bytes) else source
            ) as original:
                image = prepare_image(original.convert("RGB"))
            images.append(image)
            processor_content.append({"type": "image", "image": image})
        else:
            raise ValueError("content items must have type text or image")
        content_types.append(item_type)
    if not processor_content:
        raise ValueError("content must not be empty")
    return processor_content, images, content_types


def wait_for_gpu(device, min_free_gib, timeout=120):
    """Wait for sufficient free CUDA memory before loading the model."""
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError("minimum free CUDA memory must be finite and nonnegative")
    deadline = time.monotonic() + timeout
    minimum = min_free_gib * 2**30
    while True:
        free, total = torch.cuda.mem_get_info(device)
        if total < minimum:
            raise RuntimeError(
                f"{device} has {total / 2**30:.2f} GiB total, below the "
                f"{min_free_gib:.2f} GiB threshold. Use --min-free-gib to override."
            )
        if free >= minimum:
            print(
                f"{device} ready: {free / 2**30:.2f}/{total / 2**30:.2f} GiB free "
                f"(minimum {min_free_gib:.2f} GiB)",
                flush=True,
            )
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(
                f"{device} did not reach {min_free_gib:.2f} GiB free within {timeout}s"
            )
        time.sleep(min(2, remaining))


class WePMRuntime:
    """Qwen3.5 + MGVB + independent E/J slots and trained task heads.

    Each call accepts one ordered text/image request. Generation uses a
    KV/recurrent cache and no slot. E reads the embedding slot and applies the normalized
    projection; J reads its own slot and applies the direct candidate-index head.
    """

    def _stop_token_ids(self) -> set[int]:
        stop_ids = set()
        for value in (
            self.config.get("eos_token_id"),
            self.processor.tokenizer.eos_token_id,
        ):
            if isinstance(value, list):
                stop_ids.update(value)
            elif value is not None:
                stop_ids.add(value)
        return stop_ids

    def _load_task_modules(self, runtime_sources, auxiliary):
        c = self.config
        self.processor = AutoProcessor.from_pretrained(
            self.model_dir, local_files_only=True, trust_remote_code=False
        )
        if not self.processor.chat_template:
            raise ValueError("missing saved training chat template")
        self.processor.tokenizer.chat_template = self.processor.chat_template
        self.slot_token_id = self.processor.tokenizer.pad_token_id
        if self.slot_token_id is None:
            self.slot_token_id = self.processor.tokenizer.eos_token_id
        self.slots = {
            role: auxiliary.pop(f"unimodel.slot_pooler.{role}_slot").to(self.dtype)
            for role in ("embedding", "decision")
        }
        self.head = auxiliary.pop("unimodel.embedding_head.weight").to(self.dtype)
        self.decision_head = auxiliary.pop("unimodel.decision_head.output.weight").to(
            self.dtype
        )
        hidden = c["text_config"]["hidden_size"]
        if any(tuple(slot.shape) != (hidden,) for slot in self.slots.values()) or tuple(
            self.head.shape
        ) != (
            c["unimodel_embedding_dim"],
            hidden,
        ):
            raise ValueError("slot/head shape mismatch")
        if tuple(self.decision_head.shape) != (
            self.decision_config.output_size,
            hidden,
        ):
            raise ValueError("direct decision head shape mismatch")
        cls, self.get_rope_index = load_runtime_sources(runtime_sources)
        self.mgvb = cls(
            hidden,
            adapter_dim=c["unimodel_mgvb_adapter_dim"],
            token_budget=c["unimodel_mgvb_token_budget"],
            global_tokens=c["unimodel_mgvb_global_tokens"],
            region_tokens=c["unimodel_mgvb_region_tokens"],
            num_global_heads=c["unimodel_mgvb_num_global_heads"],
            device=self.device,
            dtype=self.dtype,
        )
        prefix = "unimodel.mgvb."
        if any(not k.startswith(prefix) for k in auxiliary):
            raise ValueError(f"unknown auxiliary weights: {sorted(auxiliary)}")
        self.mgvb.load_state_dict(
            {k.removeprefix(prefix): v for k, v in auxiliary.items()}, strict=True
        )
        self.mgvb.eval()

    def _compact_image_tokens(self, ids, grids):
        """Reserve exactly min(native length, MGVB budget) placeholders per image."""
        merge = self.config["vision_config"]["spatial_merge_size"]
        image_id = self.config["image_token_id"]
        native_lengths = (grids.prod(-1) // merge**2).tolist()
        compact_lengths = [min(int(n), self.mgvb.token_budget) for n in native_lengths]
        indices = torch.nonzero(ids[0] == image_id).flatten()
        spans = torch.split(indices, [int(n) for n in native_lengths])
        keep = torch.ones(ids.shape[1], dtype=torch.bool, device=self.device)
        for span, count in zip(spans, compact_lengths, strict=True):
            if not torch.all(torch.diff(span) == 1):
                raise ValueError("non-contiguous image placeholders")
            keep[span[count:]] = False
        return ids[:, keep], native_lengths, compact_lengths

    def _text_query(self, ids, embeddings, slot_role):
        """Build the training MGVB mean+last query from non-special prompt tokens."""
        mask = torch.ones_like(ids, dtype=torch.bool)
        for token_id in self.processor.tokenizer.all_special_ids:
            mask &= ids != token_id
        if slot_role is not None:
            mask[:, -1] = False
        count = mask.sum(dim=1)
        if not int(count[0]):
            return None
        mean = (embeddings * mask.unsqueeze(-1)).sum(dim=1) / count.to(
            self.dtype
        ).unsqueeze(-1)
        last = embeddings[:, torch.nonzero(mask[0]).flatten()[-1], :]
        return torch.cat([mean, last], dim=-1)[0]

    def _encode_images(self, pixels, grids, query):
        """Run native ViT/merger, then the learned MGVB in image order."""
        pixels = pixels.to(self.device, self.dtype)
        patches = torch.split(pixels, [int(n) for n in grids.prod(-1).tolist()])
        native_features = [
            self._vision_features(patch, grid.unsqueeze(0))
            for patch, grid in zip(patches, grids, strict=True)
        ]
        return self.mgvb(
            native_features,
            image_grid_thw=grids,
            query_embeddings=[query] * len(native_features),
            spatial_merge_size=self.config["vision_config"]["spatial_merge_size"],
        )

    @torch.inference_mode()
    def _prepare_inputs(self, content, *, task: str):
        if task not in {"generate", "embedding", "decision"}:
            raise ValueError(f"unknown task: {task}")
        generation = task == "generate"
        slot_role = None if generation else task
        processor_content, loaded_images, content_types = prepare_content(
            content, self.contracts.image_geometry.prepare_model_image
        )
        rendered = self.processor.apply_chat_template(
            [{"role": "user", "content": processor_content}],
            tokenize=False,
            add_generation_prompt=generation,
            enable_thinking=False,
        )
        encoded = self.processor(
            text=[rendered], images=loaded_images or None, return_tensors="pt"
        )
        ids = encoded["input_ids"].to(self.device)
        if ids.shape[0] != 1:
            raise ValueError("only unpadded single-request inference is supported")
        if slot_role is not None:
            ids = torch.cat(
                [ids, torch.tensor([[self.slot_token_id]], device=self.device)], dim=1
            )
        grids = encoded.get("image_grid_thw")
        positions, native_lengths, compact_lengths = [], [], []
        if grids is not None:
            grids = grids.to(self.device)
            ids, native_lengths, compact_lengths = self._compact_image_tokens(
                ids, grids
            )
        embeddings = self._token_embeddings(ids)
        # Only the appended position is a soft slot. Literal pad tokens in
        # the prompt and generated tokens always retain native embeddings.
        if slot_role is not None:
            embeddings[:, -1, :] = self.slots[slot_role]
        if grids is not None:
            compressed = self._encode_images(
                encoded["pixel_values"],
                grids,
                self._text_query(ids, embeddings, slot_role),
            )
            positions = compressed.positions
            features = torch.cat(compressed.embeddings)
            mask = ids[0] == self.config["image_token_id"]
            if int(mask.sum()) != features.shape[0]:
                raise ValueError("MGVB image-feature/placeholder mismatch")
            embeddings[0, mask, :] = features.to(self.dtype)
        rope, delta = self.get_rope_index(
            spatial_merge_size=self.config["vision_config"]["spatial_merge_size"],
            image_token_id=self.config["image_token_id"],
            video_token_id=self.config["video_token_id"],
            vision_start_token_id=self.config["vision_start_token_id"],
            input_ids=ids,
            image_grid_thw=grids,
            image_position_ids=positions if grids is not None else None,
            attention_mask=torch.ones_like(ids),
        )
        text_positions = torch.arange(ids.shape[1], device=self.device).view(1, 1, -1)
        position_ids = torch.cat([text_positions, rope], dim=0)
        metadata = {
            "input_tokens": ids.shape[1],
            "native_visual_tokens": native_lengths,
            "compact_visual_tokens": compact_lengths,
            "slot_positions": [ids.shape[1] - 1] if slot_role is not None else [],
            "slot_role": slot_role,
            "mrope_delta": int(delta[0, 0]),
            "generation_prompt": generation,
            "content_types": content_types,
        }
        return ids, embeddings, position_ids, metadata

    @torch.inference_mode()
    def embed(self, content, *, dims=None):
        """Return normalized embedding vectors at the requested MRL dimensions."""
        start = time.monotonic()
        _, embeddings, positions, metadata = self._prepare_inputs(
            content, task="embedding"
        )
        hidden = self._last_hidden(embeddings, positions)
        # Training projects in the model dtype before FP32 L2 normalization.
        projected = F.linear(hidden, self.head).float()
        if not torch.isfinite(projected).all() or projected.norm() == 0:
            raise ValueError("nonfinite or zero embedding")
        dims = dims or self.config["unimodel_embedding_mrl_dims"]
        allowed = set(self.config["unimodel_embedding_mrl_dims"])
        if any(d not in allowed for d in dims):
            raise ValueError(f"dimensions must be chosen from {sorted(allowed)}")
        vectors = {
            str(dim): F.normalize(projected[:, :dim], dim=-1)[0].cpu().tolist()
            for dim in dims
        }
        return {
            "embeddings": vectors,
            "seconds": time.monotonic() - start,
            "input": metadata,
        }

    @torch.inference_mode()
    def decide(self, request, images=()):
        """Evaluate one SystemOne choice/score/noul request using the J slot."""
        start = time.monotonic()
        protocol = self.contracts.decision_protocol
        value = protocol.request_input(request)
        count = len(value.ids)
        if count > self.decision_config.max_candidates:
            raise ValueError("candidate count exceeds the trained direct head capacity")
        wire_images = protocol.request_images(request)
        if images and wire_images:
            raise ValueError("use either local --image inputs or JSON image data URLs")
        if len(images) > 8 or sum(map(len, wire_images)) > 64 * 1024 * 1024:
            raise ValueError(
                "decision supports at most eight images and 64 MiB of image bytes"
            )
        decision_content = [
            {
                "type": "image",
                "path" if not isinstance(image, bytes) else "bytes": image,
            }
            for image in (images or wire_images)
        ]
        decision_content.append(
            {"type": "text", "text": protocol.render_input(value)[0]}
        )
        _, embeddings, positions, metadata = self._prepare_inputs(
            decision_content, task="decision"
        )
        hidden = self._last_hidden(embeddings, positions)
        # The direct head maps h_J to option indices. E normalization and
        # physical padding rows must never enter the candidate distribution.
        logits = F.linear(hidden, self.decision_head).float()[0, :count]
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite decision logits")
        probabilities = logits.softmax(dim=-1).cpu().tolist()
        return {
            **protocol.response_from_probabilities(value, probabilities),
            "candidate_count": count,
            "seconds": time.monotonic() - start,
            "input": metadata,
        }


class UniModelTransformers(WePMRuntime):
    """Hugging Face backbone with the trained WePM task modules."""

    def __init__(
        self,
        model_dir: str | Path,
        device="cuda:0",
        *,
        min_free_gib=None,
        trust_model_code=False,
    ):
        path = Path(model_dir).expanduser()
        checkpoint = WePMCheckpoint(
            path if path.is_dir() else resolve_model_path(model_dir)
        )
        self.model_dir = checkpoint.path
        self.config = checkpoint.config
        c = self.config
        runtime_sources = select_runtime_sources(
            self.model_dir, trust_model_code=trust_model_code
        )
        self.contracts = load_contracts(runtime_sources)
        self.decision_config = validate_architecture(c, self.contracts)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("supported inference devices are cpu and cuda")
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. Use inference/infer_cpu.py on CPU."
                )
            if min_free_gib is None:
                # BF16 checkpoint size plus room for the CUDA context, cache and activations.
                min_free_gib = checkpoint.weight_bytes / 2**30 + 2
            wait_for_gpu(self.device, min_free_gib)
        # FP32 also works on CPUs without native BF16 instructions. CUDA keeps
        # the BF16 path used by training and the verified NVIDIA example.
        self.dtype = torch.float32 if self.device.type == "cpu" else torch.bfloat16
        with checkpoint.backbone_directory() as backbone_dir:
            self.model, loading_info = Qwen3_5ForConditionalGeneration.from_pretrained(
                backbone_dir,
                dtype=self.dtype,
                device_map={"": str(self.device)},
                attn_implementation="sdpa",
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                output_loading_info=True,
            )
        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
            "conversion_errors",
        ):
            if loading_info.get(key):
                raise ValueError(f"incomplete HF load: {key}={loading_info[key]}")
        self.loading_info = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in loading_info.items()
        }
        self.model.eval()
        auxiliary = load_file(checkpoint.auxiliary_path, device=str(self.device))
        if checkpoint.indexed_auxiliary_keys and (
            auxiliary.keys() != checkpoint.indexed_auxiliary_keys
        ):
            raise ValueError(
                "training export task weights do not match the shard index"
            )
        self._load_task_modules(runtime_sources, auxiliary)

    def _token_embeddings(self, ids):
        return self.model.get_input_embeddings()(ids)

    def _vision_features(self, pixels, grid):
        visual = self.model.model.get_image_features(pixels, grid, return_dict=True)
        return visual.pooler_output[0]

    def _last_hidden(self, embeddings, positions):
        result = self.model.model.language_model(
            inputs_embeds=embeddings,
            position_ids=positions,
            use_cache=False,
            return_dict=True,
        )
        return result.last_hidden_state[:, -1, :]

    @torch.inference_mode()
    def generate(self, content, max_new_tokens=64):
        """Greedy generation using the Transformers text model's native cache."""
        if not 0 < max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be in [1, 512]")
        start = time.monotonic()
        ids, embeddings, positions, metadata = self._prepare_inputs(
            content, task="generate"
        )
        result = self.model.model.language_model(
            inputs_embeds=embeddings,
            position_ids=positions,
            use_cache=True,
            return_dict=True,
        )
        cache = result.past_key_values
        hidden = result.last_hidden_state[:, -1:, :]
        rope_start = int(positions[1:].max()) + 1
        stop_ids = self._stop_token_ids()
        generated = []
        finish = "length"
        for index in range(max_new_tokens):
            logits = self.model.lm_head(hidden)[:, -1, :].float()
            if not torch.isfinite(logits).all():
                raise ValueError("nonfinite generation logits")
            token = logits.argmax(dim=-1)
            generated.append(int(token[0]))
            if int(token[0]) in stop_ids:
                finish = "eos"
                break
            if index + 1 == max_new_tokens:
                break
            pos = torch.full(
                (4, 1, 1), rope_start + index, dtype=torch.long, device=self.device
            )
            pos[0] = ids.shape[1] + index
            result = self.model.model.language_model(
                input_ids=token[:, None],
                position_ids=pos,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            hidden, cache = result.last_hidden_state, result.past_key_values
        return {
            "text": self.processor.tokenizer.decode(
                generated, skip_special_tokens=True
            ),
            "token_ids": generated,
            "finish_reason": finish,
            "seconds": time.monotonic() - start,
            "input": metadata,
        }
