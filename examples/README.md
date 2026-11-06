# WePM Inference Examples

The runnable examples are grouped by device. Each device directory contains generation, embedding, and structured-decision tasks.

```text
examples/
├── gpu/
│   ├── assets/
│   ├── generation/
│   ├── embedding/
│   └── decision/
└── cpu/
    ├── assets/
    ├── generation/
    ├── embedding/
    └── decision/
```

The generation examples cover all supported input layouts:

- `single_text`: one text item
- `single_image`: one image item
- `interleaved`: text → image → text → image → text

The original HEIC images are preserved byte-for-byte in each device's `assets/` directory. The inference dependencies include `pillow-heif` for decoding them.

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

The scripts use `checkpoints/huggingface` by default. To use another complete checkpoint, set:

```bash
export WEPM_MODEL=/path/to/WePM-HF
```

GPU selection, CPU threads, and output location can be controlled with `CUDA_VISIBLE_DEVICES`, `WEPM_THREADS`, and `WEPM_OUTPUT_DIR` respectively. Results are written to timestamped JSON files under each task's `results/` directory by default.
