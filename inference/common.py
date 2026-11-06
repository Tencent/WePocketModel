"""File utilities and standalone imports of the checkpoint's model code."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_contracts(source: Path):
    """Load the checkpoint's own task routing, decision protocol and geometry."""
    name = (
        "_wepm_nvidia_"
        + hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:16]
    )
    package = types.ModuleType(name)
    package.__path__ = [str(source)]
    sys.modules[name] = package
    modules = {
        key: load_file(f"{name}.{key}", source / f"{key}.py")
        for key in (
            "generation_slots",
            "decision_config",
            "decision_protocol",
            "image_geometry",
        )
    }
    return types.SimpleNamespace(**modules)


def validate_architecture(config, contracts):
    """Accept the formal task_slots_v1 direct-J architecture, never legacy slots."""
    if config.get("unimodel_decision_enable") is not True:
        raise ValueError("this runtime requires the new JEV task_slots_v1 checkpoint")
    contracts.decision_config.validate_task_slots(config, checkpoint=True)
    decision = contracts.decision_config.DecisionConfig.from_config(config)
    if decision.head_type != "direct":
        raise ValueError("the NVIDIA example requires the trained direct decision head")
    if (
        config.get("unimodel_decision_input_profile")
        != contracts.decision_protocol.INPUT_PROFILE
    ):
        raise ValueError("checkpoint decision input protocol is incompatible")
    if not config.get("unimodel_slot_enable") or not config.get("unimodel_mgvb_enable"):
        raise ValueError("the formal model requires E/J slots and MGVB")
    if config.get("unimodel_ds_lora_enable") or config.get("unimodel_mgve_enable"):
        raise ValueError("DS-LoRA/MGVE are not part of this architecture")
    return decision


def load_runtime_sources(source: Path):
    """Reuse canonical math without importing Megatron, VeRL, or XPU packages."""
    geometry = load_file("_unimodel_hf_geometry", source / "mgvb_utils.py")
    tree = ast.parse((source / "mgvb.py").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == "mbridge.models.qwen3_5.mgvb_utils"
        ):
            node.module = geometry.__name__
    module = types.ModuleType("_unimodel_hf_mgvb")
    sys.modules[module.__name__] = module
    # Execute the frozen module after replacing its Megatron-dependent import.
    exec(compile(tree, str(source / "mgvb.py"), "exec"), module.__dict__)  # noqa: S102

    # The standalone function only needs torch; the original file's other
    # imports initialize Megatron and transformer-engine unnecessarily.
    rope_tree = ast.parse((source / "rope_utils.py").read_text())
    function = next(
        n
        for n in rope_tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "get_rope_index"
    )
    standalone = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function,
        ],
        type_ignores=[],
    )
    import torch

    namespace = {"torch": torch}
    exec(  # noqa: S102 -- import the canonical RoPE function without Megatron
        compile(
            ast.fix_missing_locations(standalone), str(source / "rope_utils.py"), "exec"
        ),
        namespace,
    )
    return module.MultiGranularityVisualBottleneck, namespace["get_rope_index"]
