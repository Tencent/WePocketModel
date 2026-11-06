# WePM Inference

This directory contains the complete Hugging Face Transformers inference implementation for WePM `task_slots_v1`. It loads the language and vision backbone, MGVB, independent embedding/decision slots, embedding head, and structured-decision head.

Supported tasks:

- `generate`: text or multimodal generation
- `embed`: L2-normalized multimodal embeddings
- `decision`: structured choice, score, and binary judgment

## Installation

Create a Python 3.10–3.13 environment and install the PyTorch build for the target device.

NVIDIA GPU (CUDA 12.8 example):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r inference/requirements.txt
```

CPU:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r inference/requirements.txt
```

`Pillow` handles standard images and `pillow-heif` decodes the unmodified HEIC files bundled with the examples.

## Model directory

The default examples use `checkpoints/huggingface`. A complete model directory contains:

```text
huggingface/
├── config.json
├── model-*.safetensors
├── model.safetensors.index.json
├── unimodel_aux.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── processor_config.json
├── chat_template.jinja
├── runtime_sources/
└── export_complete.json
```

Do not remove `runtime_sources/` or `unimodel_aux.safetensors`; they are part of the checkpoint contract.

## Simple input

Use `--prompt` for one text item and repeat `--image` for one or more images. In this compact form, images are placed before the text.

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
  --model checkpoints/huggingface --task generate \
  --prompt "Describe this image." --image /path/to/image.png \
  --output results/generation.json
```

For CPU, use `inference/infer_cpu.py` and add `--threads 8`.

## Ordered multimodal input

Use `--content-json` when text and images must retain an explicit order. Relative image paths are resolved from the JSON file's directory.

```json
{
  "content": [
    {"type": "text", "text": "Inspect the first image:"},
    {"type": "image", "path": "../images/first.heic"},
    {"type": "text", "text": "Compare it with the second image:"},
    {"type": "image", "path": "../images/second.heic"},
    {"type": "text", "text": "Summarize the differences."}
  ]
}
```

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
  --model checkpoints/huggingface --task generate \
  --content-json /path/to/content.json \
  --output results/interleaved.json
```

`--content-json` cannot be combined with `--prompt` or `--image`.

## Structured decision

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
  --model checkpoints/huggingface --task decision \
  --decision-json examples/gpu/demo_data/decision/request.json \
  --output results/decision.json
```

The decision request uses `state` for the content being evaluated and `questions.q` for the task definition. Supported types are `choice`, `score`, and `noul`.

## Main arguments

| Argument | Description |
| --- | --- |
| `--model` | Complete local WePM Hugging Face directory |
| `--task` | `generate`, `embed`, `both`, or `decision` |
| `--prompt` | Simple text input |
| `--image` | Simple local image input; may be repeated |
| `--content-json` | Ordered text/image content for generation or embedding |
| `--decision-json` | Structured decision request; required only for `decision` |
| `--max-new-tokens` | Generation length from 1 to 512 |
| `--embedding-dim` | Embedding dimension; default 2048 |
| `--threads` | CPU compute threads; default 8 |
| `--timeout` | Maximum runtime in seconds; default 1200 |
| `--output` | New JSON output path; required and must not already exist |

Each invocation processes one request, writes a JSON result, releases the model, and exits. See [`examples/`](../examples) for complete GPU and CPU cases.
