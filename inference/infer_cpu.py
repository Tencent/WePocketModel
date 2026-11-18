#!/usr/bin/env python3
"""Run Transformers or GGUF WePM inference on CPU, then exit."""

import os


def main():
    # Set visibility before importing PyTorch or Transformers. The CPU path
    # neither waits for the GPU keeper nor initializes a CUDA context.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    from infer import main as run_inference

    run_inference(device="cpu")


if __name__ == "__main__":
    main()
