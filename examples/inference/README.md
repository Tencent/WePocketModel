# WePM Inference Examples

These examples run the complete `task_slots_v1` WePM checkpoint with Hugging Face Transformers. They support text and image inputs across three capability families:

- `generate`: text or image-conditioned generation
- `embed`: L2-normalized multimodal embeddings
- `decision`: structured choice, score, and binary judgment

Each invocation processes one request, writes a JSON result, releases the model, and exits.

## Requirements

The examples target Linux and Python 3.10–3.13. Create a virtual environment first:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install PyTorch for the target device.

NVIDIA GPU (CUDA 12.8 example):

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

CPU:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For another CUDA version, select the matching PyTorch and torchvision wheels from the [PyTorch installation guide](https://pytorch.org/get-started/previous-versions/).

## Model directory

Set the path to the complete Hugging Face checkpoint:

```bash
export WEPM_MODEL=/path/to/WePM-HF
```

The directory must contain the model shards and all custom WePM assets:

```text
WePM-HF/
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

Do not remove `runtime_sources/` or `unimodel_aux.safetensors`: they contain the checkpoint-aligned MGVB implementation, task slots, embedding head, and decision head.

## NVIDIA GPU

The GPU entry uses BF16 on one CUDA device. At least 16 GiB of GPU memory is recommended.

Text generation:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$WEPM_MODEL" --task generate \
  --prompt 'Describe WePM in one sentence.' \
  --max-new-tokens 64 --output results/generation.json
```

Image-conditioned generation:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$WEPM_MODEL" --task generate \
  --prompt 'Describe this image.' --image /path/to/image.png \
  --output results/image-generation.json
```

## CPU

The CPU entry disables CUDA before importing PyTorch and loads the same safetensors checkpoint in FP32. Model weights require approximately 17 GiB of system memory; at least 24 GiB total memory is recommended.

```bash
python infer_cpu.py \
  --model "$WEPM_MODEL" --task generate \
  --prompt 'Describe WePM in one sentence.' \
  --threads 8 --max-new-tokens 32 \
  --output results/cpu-generation.json
```

## Embeddings

Text embedding on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$WEPM_MODEL" --task embed \
  --prompt 'A cat is sitting by the window.' \
  --embedding-dim 1024 --output results/text-embedding.json
```

For CPU, replace `infer.py` with `infer_cpu.py` and add `--threads 8`.

Image embedding accepts one or more `--image` arguments. Text and images can also be supplied together. Supported dimensions are 64, 128, 256, 512, 1024, and 2048. The vectors are L2-normalized and stored under `embedding.embeddings`.

## Structured decision

Run the bundled choice example on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python infer.py \
  --model "$WEPM_MODEL" --task decision \
  --decision-json examples/decision.json \
  --output results/decision.json
```

For CPU, replace `infer.py` with `infer_cpu.py` and add `--threads 8`.

The request uses `state` for the content being evaluated and `questions.q` for the task definition:

- `choice`: `criteria` maps ordered candidate IDs to descriptions.
- `score`: `criteria` is an ordered list of score descriptions.
- `noul`: no `criteria` is provided; the result is the probability that the statement holds.

Local decision images can be supplied with repeated `--image` arguments. Alternatively, the request JSON may contain base64 image data URLs. Do not use both forms in one request.

## Main arguments

| Argument | Description |
| --- | --- |
| `--model` | Complete local WePM Hugging Face directory |
| `--task` | `generate`, `embed`, `both`, or `decision` |
| `--prompt` | Text input |
| `--image` | Local image path; may be repeated |
| `--decision-json` | Structured decision request; required only for `decision` |
| `--max-new-tokens` | Generation length from 1 to 512 |
| `--embedding-dim` | Embedding dimension; default 2048 |
| `--threads` | CPU compute threads; default 8 |
| `--timeout` | Maximum runtime in seconds; default 1200 |
| `--output` | New JSON output path; required and must not already exist |

Generation uses greedy decoding and each process handles one request at a time.
