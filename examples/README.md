# WePM Inference Examples

The runnable examples are grouped by device. Each device directory contains generation, embedding, and structured-decision tasks.

```text
examples/
├── gpu/
│   ├── demo_data/
│   │   ├── images/
│   │   ├── generation/
│   │   └── decision/
│   ├── generation/
│   ├── embedding/
│   └── decision/
└── cpu/
    ├── demo_data/
    │   ├── images/
    │   ├── generation/
    │   └── decision/
    ├── generation/
    ├── embedding/
    └── decision/
```

Task directories contain only runnable shell scripts. All JSON requests and image inputs live under the device-specific `demo_data/` directory.

The generation examples cover all supported input layouts:

- `single_text`: one text item
- `single_image`: one image item
- `interleaved`: text → image → text → image → text

The original HEIC images are preserved byte-for-byte in each device's `demo_data/images/` directory. The inference dependencies include `pillow-heif` for decoding them.

## NVIDIA GPU

```bash
bash examples/gpu/generation/run.sh single_text
bash examples/gpu/generation/run.sh single_image
bash examples/gpu/generation/run.sh interleaved
bash examples/gpu/embedding/run.sh
bash examples/gpu/decision/run.sh
```

## CPU

```bash
bash examples/cpu/generation/run.sh single_text
bash examples/cpu/generation/run.sh single_image
bash examples/cpu/generation/run.sh interleaved
bash examples/cpu/embedding/run.sh
bash examples/cpu/decision/run.sh
```

The scripts default to `tencent/WePocketModel1.0-4B`. The first online run downloads the complete Hugging Face snapshot; later runs reuse cached files. See the [model table](../README.md#model-weights) for both formats.

To select the 2B Transformers model:

```bash
export WEPM_MODEL=tencent/WePocketModel1.0-2B
```

The shared runtime requires a compatible `task_slots_v1` checkpoint. The embedding examples explicitly request 1024 dimensions, which the chosen checkpoint must support.

For a pre-downloaded local model:

```bash
export WEPM_MODEL=/path/to/WePocketModel1.0-4B
```

Local model directories are loaded without network access. Set `HF_HUB_OFFLINE=1` to use only the existing Hub cache when `WEPM_MODEL` is a repository ID. Use the [inference CLI](../inference/README.md) for `--revision`, `--cache-dir`, and `--local-files-only`.

CPU examples also accept a local WePM GGUF bundle via `WEPM_MODEL` after [building the native library](../inference/README.md#gguf-on-cpu).

## Apple silicon (MLX)

The MLX examples are shipped inside the `-MLX` model repositories rather than in this directory, because that runtime is self-contained:

```bash
hf download tencent/WePocketModel1.0-4B-MLX --local-dir WePocketModel1.0-4B-MLX
cd WePocketModel1.0-4B-MLX
python -m pip install -r requirements.txt

python runtime/infer_mlx.py --model . --task generate --prompt "Describe WePM in one sentence." \
    --max-new-tokens 64 --output out/generation.json
python runtime/infer_mlx.py --model . --task embed --image /path/to/image.jpg \
    --embedding-dim 1024 --output out/embedding.json
python runtime/infer_mlx.py --model . --task decision --decision-json request.json \
    --output out/decision.json
```

See [MLX on Apple silicon](../README.md#mlx-on-apple-silicon) for setup and the decision request format.

GPU selection, CPU threads, and output location can be controlled with `CUDA_VISIBLE_DEVICES`, `WEPM_THREADS`, and `WEPM_OUTPUT_DIR` respectively. Set `WEPM_MIN_FREE_GIB` to override the default CUDA loading threshold of checkpoint size plus 2 GiB. Results are written to timestamped JSON files under each task's `results/` directory by default.
