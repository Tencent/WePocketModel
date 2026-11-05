"""Minimal single-GPU inference example for WePM-4B.

Usage:
    python inference.py --image ./assert/wepm_function.png \
        --prompt "Describe this figure in one sentence."
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = "./checkpoints/WePM-4B"


def load_model(model_path: str, dtype: str, device_map: str):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=getattr(torch, dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, processor


def build_inputs(processor, image_path: str, prompt: str):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


@torch.inference_mode()
def generate(model, processor, inputs, max_new_tokens: int) -> str:
    inputs = inputs.to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[-1] :]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="WePM-4B single-image inference")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="model directory or hub id")
    parser.add_argument("--image", required=True, help="path to one local image")
    parser.add_argument(
        "--prompt",
        default="Describe this image in one sentence.",
        help="user instruction paired with the image",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16"], help="compute dtype"
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='device placement, e.g. "auto" or "cuda:0"',
    )
    args = parser.parse_args()

    if not Path(args.image).is_file():
        raise FileNotFoundError(f"image not found: {args.image}")

    model, processor = load_model(args.model, args.dtype, args.device_map)
    inputs = build_inputs(processor, args.image, args.prompt)
    print(generate(model, processor, inputs, args.max_new_tokens))


if __name__ == "__main__":
    main()
