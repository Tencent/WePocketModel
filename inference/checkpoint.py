"""Locate and validate the files of a WePM release or raw training export."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

INDEX_NAME = "model.safetensors.index.json"
RELEASE_SCHEMA = "wepm-nvidia-task-slots-v1"
LLAMA_CPP_COMMIT = "37b0f9c36ae910c5a445b3934e840084ebc9b132"
BUNDLED_RUNTIME = Path(__file__).resolve().with_name("runtime_sources")
RUNTIME_FILES = (
    "generation_slots.py",
    "decision_config.py",
    "decision_protocol.py",
    "image_geometry.py",
    "mgvb.py",
    "mgvb_utils.py",
    "rope_utils.py",
)


def select_runtime_sources(model_dir: Path, *, trust_model_code: bool) -> Path:
    source = model_dir / "runtime_sources" if trust_model_code else BUNDLED_RUNTIME
    missing = [name for name in RUNTIME_FILES if not (source / name).is_file()]
    if missing:
        raise ValueError(
            f"incomplete WePM runtime at {source}; missing: {', '.join(missing)}"
        )
    return source


class WePMCheckpoint:
    def __init__(self, model_dir: str | Path):
        self.path = Path(model_dir).expanduser().resolve()
        manifest = self.path / "export_complete.json"
        self.is_release = manifest.is_file()
        if self.is_release:
            if json.loads(manifest.read_text()).get("schema") != RELEASE_SCHEMA:
                raise ValueError(
                    f"a compatible task_slots_v1 WePM export is required: {manifest}"
                )
            self.auxiliary_path = self.path / "unimodel_aux.safetensors"
            required = (
                "config.json",
                "unimodel_aux.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "processor_config.json",
                "chat_template.jinja",
            )
            missing = [name for name in required if not (self.path / name).is_file()]
            if missing:
                raise ValueError(
                    f"incomplete WePM export at {self.path}; missing: {', '.join(missing)}. "
                    "Download the complete model repository, not only the weight files."
                )
        else:
            self.auxiliary_path = self.path / "model-unimodel.safetensors"
            if not self.auxiliary_path.is_file():
                raise ValueError(
                    f"expected a WePM release or training export: {self.path}"
                )

        self.config = json.loads((self.path / "config.json").read_text())
        index = self.path / INDEX_NAME
        self.indexed_auxiliary_keys = set()
        if index.is_file():
            weight_map = json.loads(index.read_text()).get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(f"invalid weight index: {index}")
            self.backbone_weights = {
                key: name
                for key, name in weight_map.items()
                if not key.startswith("unimodel.")
            }
            self.indexed_auxiliary_keys = set(weight_map) - self.backbone_weights.keys()
            if self.is_release:
                if self.indexed_auxiliary_keys:
                    raise ValueError("a WePM release must index only backbone weights")
            elif not self.indexed_auxiliary_keys or any(
                weight_map[key] != self.auxiliary_path.name
                for key in self.indexed_auxiliary_keys
            ):
                raise ValueError(
                    "training export must index all task weights in model-unimodel.safetensors"
                )
            if not self.backbone_weights or any(
                not isinstance(name, str)
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or Path(name).suffix != ".safetensors"
                or Path(name) == Path(self.auxiliary_path.name)
                for name in self.backbone_weights.values()
            ):
                raise ValueError(f"invalid WePM backbone shard index: {index}")
            self.backbone_shards = sorted(set(self.backbone_weights.values()))
        elif self.is_release:
            self.backbone_shards = ["model.safetensors"]
        else:
            raise ValueError(f"training export requires a weight index: {index}")

        weights = [
            self.auxiliary_path,
            *(self.path / name for name in self.backbone_shards),
        ]
        for path in weights:
            if not path.is_file():
                raise ValueError(f"missing model weight file: {path}")
        self.weight_bytes = sum(path.stat().st_size for path in weights)

    @contextmanager
    def backbone_directory(self):
        """Present only backbone tensors to Transformers without copying weights."""
        if self.is_release:
            yield self.path
            return
        with TemporaryDirectory(prefix="wepm-backbone-") as directory:
            backbone = Path(directory)
            for name in ("config.json", *self.backbone_shards):
                link = backbone / name
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(self.path / name)
            generation_config = self.path / "generation_config.json"
            if generation_config.is_file():
                (backbone / generation_config.name).symlink_to(generation_config)
            (backbone / INDEX_NAME).write_text(
                json.dumps({"metadata": {}, "weight_map": self.backbone_weights})
            )
            yield backbone
