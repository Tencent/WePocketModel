#!/usr/bin/env python3
"""Finite G/E/J inference with Transformers on CPU or CUDA; save JSON and exit."""

import argparse
import gc
import json
import signal
from pathlib import Path

import torch
from common import write_json
from runtime import UniModelTransformers, wait_for_gpu


def main(*, device="cuda:0"):
    device = torch.device(device)
    is_cpu = device.type == "cpu"
    parser = argparse.ArgumentParser(description=f"{__doc__} Device: {device}.")
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="complete local Hugging Face model directory",
    )
    parser.add_argument(
        "--task", choices=["generate", "embed", "both", "decision"], default="generate"
    )
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--image", type=Path, action="append", default=[])
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
        choices=[64, 128, 256, 512, 1024, 2048],
        default=2048,
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
    if is_cpu:
        if args.threads < 1:
            parser.error("threads must be positive")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    if (args.task == "decision") != (args.decision_json is not None):
        parser.error("--decision-json is required only with --task decision")
    decision_request = (
        json.loads(args.decision_json.read_text()) if args.decision_json else None
    )
    if args.output.exists():
        raise FileExistsError(args.output)

    def timeout_handler(_signum, _frame):
        raise TimeoutError(f"inference exceeded {args.timeout} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)
    probe = None
    model = None
    try:
        if not is_cpu:
            probe = wait_for_gpu()
        backend = "hf"
        model = UniModelTransformers(args.model, device=device)
        result = {}
        if args.task in ("generate", "both"):
            result["generation"] = model.generate(
                args.prompt, args.image, args.max_new_tokens
            )
        if args.task in ("embed", "both"):
            result["embedding"] = model.embed(
                args.prompt, args.image, dims=[args.embedding_dim]
            )
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
