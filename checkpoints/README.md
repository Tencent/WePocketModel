# WePM Checkpoints

- `huggingface/`: complete Hugging Face safetensors checkpoint used by the current GPU and CPU examples.
- `gguf/`: hybrid GGUF bundle for the language backbone and multimodal/task modules. It is not an ONNX model and requires the complete bundle plus the matching native runtime.

Large weight files are stored with Git LFS. Clone with Git LFS enabled so model pointers are replaced by their binary contents.
