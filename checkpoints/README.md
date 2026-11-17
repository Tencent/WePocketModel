# WePM Checkpoints

The planned model releases are hosted separately from the GitLab inference code:

| Size | Hugging Face / safetensors |
| --- | --- |
| 4B | [tencent/WePocketModel1.0-4B](https://huggingface.co/tencent/WePocketModel1.0-4B) |
| 2B | [tencent/WePocketModel1.0-2B](https://huggingface.co/tencent/WePocketModel1.0-2B) |

The default examples resolve `tencent/WePocketModel1.0-4B` through the Hugging Face cache. Downloads require the model repositories to be published. This directory can optionally hold local downloads:

```bash
hf download tencent/WePocketModel1.0-4B --local-dir checkpoints/WePocketModel1.0-4B
hf download tencent/WePocketModel1.0-2B --local-dir checkpoints/WePocketModel1.0-2B
```

Set `WEPM_MODEL` to the resulting local directory to run without network access. Download the complete export, including `runtime_sources/`, `unimodel_aux.safetensors`, the processor and tokenizer files, and the manifest. New model subdirectories under `checkpoints/` are ignored by Git.

Model files are not tracked in this GitLab repository. Download weights from Hugging Face independently of cloning the inference code. Existing local exports are preserved on disk but excluded from Git tracking.
