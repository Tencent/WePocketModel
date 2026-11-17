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

## Model download and local directory

`--model` accepts a Hugging Face model repository ID or a complete local export directory. The default is `tencent/WePocketModel1.0-4B`; use `tencent/WePocketModel1.0-2B` to select the smaller release once published. The same loader requires a compatible `task_slots_v1` export; 2B inference must be validated after its checkpoint is available.

The first online run downloads a complete snapshot, including the auxiliary weights and runtime sources, into the standard Hugging Face cache. Later runs reuse cached files, but may check the repository for updates. Use `--revision` to pin a tag or commit, `--cache-dir` to select a cache location, and `--local-files-only` (or `HF_HUB_OFFLINE=1`) to prohibit network access. Downloads finish before the inference timeout starts and before CUDA resources are allocated. Download errors, including an unpublished repository or a missing offline cache, are not silently replaced by another model.

To download ahead of time into a local directory:

```bash
hf download tencent/WePocketModel1.0-4B --local-dir checkpoints/WePocketModel1.0-4B
```

Pass `--model checkpoints/WePocketModel1.0-4B` to load that directory without network access. The GitLab repository does not bundle model weights; download either the 2B or 4B complete Hugging Face export.

A complete local model directory contains:

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
  --model tencent/WePocketModel1.0-4B --task generate \
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
  --model tencent/WePocketModel1.0-4B --task generate \
  --content-json /path/to/content.json \
  --output results/interleaved.json
```

`--content-json` cannot be combined with `--prompt` or `--image`.

## Structured decision

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py \
  --model tencent/WePocketModel1.0-4B --task decision \
  --decision-json examples/gpu/demo_data/decision/request.json \
  --output results/decision.json
```

The decision request uses `state` for the content being evaluated and `questions.q` for the task definition. Supported types are `choice`, `score`, and `noul`.

## Main arguments

| Argument | Description |
| --- | --- |
| `--model` | Hugging Face repository ID or complete local WePM directory; default `tencent/WePocketModel1.0-4B` |
| `--revision` | Hub branch, tag, or commit; pin a commit for reproducible inference |
| `--cache-dir` | Hugging Face cache directory; otherwise use the standard Hub cache |
| `--local-files-only` | Do not access the network; fail if the requested snapshot is not cached |
| `--min-free-gib` | CUDA only: free-memory threshold before loading; default 14 GiB for 4B, not a measured 2B requirement |
| `--task` | `generate`, `embed`, `both`, or `decision` |
| `--prompt` | Simple text input |
| `--image` | Simple local image input; may be repeated |
| `--content-json` | Ordered text/image content for generation or embedding |
| `--decision-json` | Structured decision request; required only for `decision` |
| `--max-new-tokens` | Generation length from 1 to 512 |
| `--embedding-dim` | Dimension supported by the chosen checkpoint; defaults to its full embedding dimension (2048 for the current 4B export) |
| `--threads` | CPU compute threads; default 8 |
| `--timeout` | Maximum inference lifetime in seconds, excluding model download; default 1200 |
| `--output` | New JSON output path; required and must not already exist |

Each invocation processes one request, writes a JSON result, releases the model, and exits. See [`examples/`](../examples) for complete GPU and CPU cases.
