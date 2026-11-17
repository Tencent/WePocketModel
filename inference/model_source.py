"""Resolve a local WePM export or download a complete Hugging Face snapshot."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODEL = "tencent/WePocketModel1.0-4B"


def validate_model_directory(model_dir: Path) -> Path:
    """Check the export contract before allocating inference resources."""
    required = [
        "export_complete.json",
        "config.json",
        "unimodel_aux.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
    ]
    required.extend(
        f"runtime_sources/{name}.py"
        for name in (
            "generation_slots",
            "decision_config",
            "decision_protocol",
            "image_geometry",
            "mgvb",
            "mgvb_utils",
            "rope_utils",
        )
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise ValueError(
            f"incomplete WePM export at {model_dir}; missing: {', '.join(missing)}. "
            "Download the complete model repository, not only the weight files."
        )
    manifest = json.loads((model_dir / "export_complete.json").read_text())
    if manifest.get("schema") != "wepm-nvidia-task-slots-v1":
        raise ValueError("a compatible task_slots_v1 WePM export is required")
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid weight index: {index}")
        for filename in set(weight_map.values()):
            if not isinstance(filename, str) or not (model_dir / filename).is_file():
                raise ValueError(f"missing model weight shard: {filename}")
    elif not (model_dir / "model.safetensors").is_file():
        raise ValueError(f"missing model.safetensors or sharded weight index at {model_dir}")
    return model_dir


def resolve_model_path(
    model: str | Path = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> Path:
    """Local directories stay offline; repo IDs use the standard Hub cache."""
    value = str(model)
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return validate_model_directory(candidate.resolve())
    if candidate.exists() or isinstance(model, Path) or value.startswith(("/", ".", "~")):
        raise FileNotFoundError(f"local model directory not found: {candidate}")
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import validate_repo_id

    validate_repo_id(value)
    snapshot = snapshot_download(
        repo_id=value,
        revision=revision,
        cache_dir=str(Path(cache_dir).expanduser()) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )
    return validate_model_directory(Path(snapshot).resolve())
