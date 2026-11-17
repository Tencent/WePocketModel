#!/usr/bin/env python3
"""Finite G/E/J inference with Transformers on CPU or CUDA; save JSON and exit."""

import argparse
import gc
import json
import signal
from pathlib import Path

import torch
from common import write_json
from model_source import DEFAULT_MODEL, resolve_model_path
from runtime import UniModelTransformers, wait_for_gpu


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
        help="Hugging Face model repo ID or complete local WePM export directory",
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
            default=14,
            help="required free CUDA memory before loading (default: 14 GiB for 4B)",
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
        "--embedding-dim",
        type=int,
        help="embedding dimension supported by the checkpoint (default: its full dimension)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="Maximum inference lifetime in seconds (1-1800)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or not 1 <= args.timeout <= 1800:
        parser.error("max-new-tokens must be in [1, 512] and timeout in [1, 1800]")
    if not is_cpu and not 0 < args.min_free_gib < float("inf"):
        parser.error("min-free-gib must be finite and positive")
    if args.embedding_dim is not None and args.embedding_dim < 1:
        parser.error("embedding-dim must be positive")
    if is_cpu:
        if args.threads < 1:
            parser.error("threads must be positive")
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
    if args.output.exists():
        raise FileExistsError(args.output)
    model_dir = resolve_model_path(
        args.model,
        revision=args.revision,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"inference exceeded {args.timeout} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    probe = None
    model = None
    try:
        if not is_cpu:
            probe = wait_for_gpu(min_free_gib=args.min_free_gib)
        backend = "hf"
        model = UniModelTransformers(model_dir, device=device)
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
                    "language_precision": str(model.dtype),
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
        del model, probe
        gc.collect()
        if not is_cpu:
            torch.cuda.empty_cache()
        signal.alarm(0)


if __name__ == "__main__":
    main()
