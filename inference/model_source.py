"""Resolve a local WePM checkpoint or download a complete Hub snapshot."""

from __future__ import annotations

from pathlib import Path

from checkpoint import WePMCheckpoint

DEFAULT_MODEL = "tencent/WePocketModel1.0-4B"


def validate_model_directory(model_dir: Path) -> Path:
    """Validate Transformers checkpoints before allocating model resources."""
    if not (model_dir / "bundle.json").is_file():
        WePMCheckpoint(model_dir)
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
