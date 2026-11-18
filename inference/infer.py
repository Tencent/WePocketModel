#!/usr/bin/env python3
"""Run G/E/J inference with Transformers or CPU GGUF and write a JSON result."""

import argparse
import gc
import json
import math
import signal
from pathlib import Path

import torch
from common import write_json
from model_source import DEFAULT_MODEL, resolve_model_path
from runtime import UniModelTransformers


def load_content(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    content = payload.get("content") if isinstance(payload, dict) else None
    if not isinstance(content, list) or not content:
        raise ValueError(f"{path} must contain a nonempty content array")
    resolved = []
    for index, item in enumerate(content):
        if not isinstance(item, dict) or item.get("type") not in {"text", "image"}:
            raise ValueError(f"content[{index}] must be a text or image object")
        if item["type"] == "text":
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"content[{index}].text must be nonempty")
            resolved.append({"type": "text", "text": text})
            continue
        value = item.get("path")
        if not isinstance(value, str) or not value:
            raise ValueError(f"content[{index}].path must be nonempty")
        image = Path(value).expanduser()
        if not image.is_absolute():
            image = path.parent / image
        image = image.resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        resolved.append({"type": "image", "path": image})
    return resolved


def main(*, device="cuda:0"):
    device = torch.device(device)
    is_cpu = device.type == "cpu"
    parser = argparse.ArgumentParser(description=f"{__doc__} Device: {device}.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Hugging Face model repo ID, local WePM export, or CPU GGUF bundle",
    )
    parser.add_argument(
        "--trust-model-code",
        action="store_true",
        help="execute the selected model's runtime_sources/ Python code with your user permissions",
    )
    parser.add_argument("--revision", help="Hub branch, tag or commit to download")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face download cache")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="use cached/local files without network",
    )
    if not is_cpu:
        parser.add_argument(
            "--min-free-gib",
            type=float,
            help="required free CUDA memory (default: checkpoint size plus 2 GiB)",
        )
    parser.add_argument(
        "--task", choices=["generate", "embed", "both", "decision"], default="generate"
    )
    parser.add_argument("--prompt", help="text input; images are placed before it")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument(
        "--content-json",
        type=Path,
        help="ordered content array for text/image interleaving",
    )
    parser.add_argument(
        "--decision-json",
        type=Path,
        help="SystemOne request with state and questions.q",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32 if is_cpu else 64)
    if is_cpu:
        parser.add_argument(
            "--threads", type=int, default=8, help="CPU compute threads (default: 8)"
        )
        parser.add_argument(
            "--gguf-variant",
            choices=["q4", "bf16"],
            help="GGUF language weights (default: q4)",
        )
        parser.add_argument("--gguf-library", type=Path, help="path to libwepm_gguf")
        parser.add_argument(
            "--context-size", type=int, help="GGUF context size (default: 4096)"
        )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        help="embedding dimension supported by the checkpoint (default: its full dimension)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Request interruption after 1-1800 seconds, excluding download (default: 1200). "
        "Native calls may delay interruption until they return.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or not 1 <= args.timeout <= 1800:
        parser.error("max-new-tokens must be in [1, 512] and timeout in [1, 1800]")
    if (
        not is_cpu
        and args.min_free_gib is not None
        and (not math.isfinite(args.min_free_gib) or args.min_free_gib < 0)
    ):
        parser.error("min-free-gib must be finite and nonnegative")
    if args.embedding_dim is not None and args.embedding_dim < 1:
        parser.error("embedding-dim must be positive")
    if is_cpu:
        if args.threads < 1:
            parser.error("threads must be positive")
        if args.context_size is not None and args.context_size < 128:
            parser.error("context-size must be at least 128")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    if (args.task == "decision") != (args.decision_json is not None):
        parser.error("--decision-json is required only with --task decision")
    if args.content_json and (args.prompt is not None or args.image):
        parser.error("--content-json cannot be combined with --prompt or --image")
    if args.task == "decision" and args.content_json:
        parser.error("decision input is defined by --decision-json")
    decision_request = (
        json.loads(args.decision_json.read_text(encoding="utf-8"))
        if args.decision_json
        else None
    )
    if args.content_json:
        content = load_content(args.content_json.resolve())
    else:
        content = [{"type": "image", "path": image.resolve()} for image in args.image]
        prompt = args.prompt
        if prompt is None and not content and args.task != "decision":
            prompt = "请用一句话介绍你自己。"
        if prompt:
            content.append({"type": "text", "text": prompt})
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    model_dir = resolve_model_path(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    backend = "gguf" if (model_dir / "bundle.json").is_file() else "hf"
    if backend == "gguf" and not is_cpu:
        parser.error("the GGUF backend runs on CPU; use inference/infer_cpu.py")
    if (
        is_cpu
        and backend == "hf"
        and any(
            value is not None
            for value in (args.gguf_variant, args.gguf_library, args.context_size)
        )
    ):
        parser.error("GGUF options require a GGUF bundle")

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"inference exceeded {args.timeout} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    model = None
    try:
        if backend == "gguf":
            from gguf_runtime import UniModelGGUF

            language_precision = args.gguf_variant or "q4"
            model = UniModelGGUF(
                model_dir,
                variant=language_precision,
                library=args.gguf_library,
                context_size=args.context_size or 4096,
                threads=args.threads,
                trust_model_code=args.trust_model_code,
            )
        else:
            model = UniModelTransformers(
                model_dir,
                device=device,
                min_free_gib=None if is_cpu else args.min_free_gib,
                trust_model_code=args.trust_model_code,
            )
            language_precision = str(model.dtype)
        result = {}
        if args.task in ("generate", "both"):
            result["generation"] = model.generate(content, args.max_new_tokens)
        if args.task in ("embed", "both"):
            embedding_dim = (
                args.embedding_dim
                if args.embedding_dim is not None
                else model.config["unimodel_embedding_dim"]
            )
            result["embedding"] = model.embed(content, dims=[embedding_dim])
        if args.task == "decision":
            result["decision"] = model.decide(decision_request, args.image)
        write_json(args.output, result)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "device": str(model.device),
                    "backend": backend,
                    "language_precision": language_precision,
                    "dtype": str(model.dtype),
                    "generation": result.get("generation", {}).get("text"),
                    "decision": result.get("decision", {}).get("answers"),
                },
                ensure_ascii=False,
            )
        )
    finally:
        if model is not None and hasattr(model, "close"):
            model.close()
        del model
        gc.collect()
        if not is_cpu:
            torch.cuda.empty_cache()
        signal.alarm(0)


if __name__ == "__main__":
    main()
