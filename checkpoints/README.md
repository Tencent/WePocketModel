# WePM Checkpoints

The 2B and 4B models each have a Transformers export, a GGUF bundle and an MLX repository. The [model table](../README.md#model-weights) lists all six Hugging Face repositories.

Use `hf download` to download a complete snapshot into a local directory. For example, choose one of the following 4B formats.

```bash
hf download tencent/WePocketModel1.0-4B --local-dir checkpoints/WePocketModel1.0-4B
hf download tencent/WePocketModel1.0-4B-GGUF --local-dir checkpoints/WePocketModel1.0-4B-GGUF
hf download tencent/WePocketModel1.0-4B-MLX --local-dir checkpoints/WePocketModel1.0-4B-MLX
```

Use the corresponding 2B repository ID and destination directory for 2B. Set `WEPM_MODEL` to the directory you downloaded, or pass it to the inference CLI with `--model`. Use the CPU entry point for a GGUF bundle. An MLX snapshot carries its own `runtime/` directory and requirements; see [MLX on Apple silicon](../README.md#mlx-on-apple-silicon). See the [inference guide](../inference/README.md) for setup and required files.

Git ignores local download directories under `checkpoints/`.
