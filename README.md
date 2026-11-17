<h1 align="center">WePM: Hierarchical Conflict Reconciliation for Unified MLLMs</h1>

<p align="center">
  <i>Unified Pocket Models for In-App Multimodal Intelligence</i>
</p>

<p align="center">
  <b>Model weights</b>: TBD &nbsp;|&nbsp; <b>Technical Report</b>: TBD &nbsp;|&nbsp; <b>Code & Evaluation</b>: TBD
</p>

## Model Introduction

On-device AI applications rarely depend on a single capability. A personal assistant may need multimodal retrieval to find local content, generation to understand and respond, and decision making to route the next action. Deploying a separate specialist for each path duplicates vision encoders and language backbones, increasing resident memory, system complexity and the cost of keeping intelligence always available on device.

**WePM** is a 4B unified multimodal model designed around a different principle: one compact vision–language backbone should support multiple capability families without sacrificing their individual quality. A single set of parameters provides:

- **Representation** — L2-normalized multimodal embeddings for image retrieval, text retrieval and candidate ranking.
- **Generation** — OCR, image captioning, visual question answering, summarization and assistant responses.
- **Decision** — typed outputs for option selection, ordinal scoring and binary judgment.

In BF16, the shared 4B backbone occupies approximately 8 GB of weight memory. More importantly, it replaces several independently resident models with one common foundation for local multimodal applications.

<p align="center">
  <img src="assert/wepm_function.png" width="640" alt="WePM capabilities and applications">
</p>

### Reconciling Conflicts in a Unified Model

Simply mixing heterogeneous tasks does not make them compatible. WePM identifies conflicts at three levels: different tasks require different visual granularity, their datasets provide different forms of supervision, and their losses operate at different scales and training rhythms. WePM reconciles these conflicts with three complementary components:

- **MGVB (Multi-Granularity Visual Bottleneck)** organizes a fixed visual-token budget into global, regional and fine-grained banks. It preserves scene-level semantics, object relations and local details while allowing each task to focus on the evidence it needs.
- **ULKD (Unified Listwise Knowledge Distillation)** transfers complementary knowledge from generation and representation teachers into one student. On the generation path, the teacher supervises prefixes visited by the student itself; on the representation path, the student learns both candidate rankings and embedding structure.
- **SEMO (Stable and Efficient Multitask Optimization)** schedules heterogeneous tasks and accumulates their gradients through sequential forward/backward passes before a shared update. This stabilizes optimization while avoiding the memory cost of retaining every task graph concurrently.

Together, these designs align visual evidence, learning signals and optimization dynamics within one compact backbone.

<p align="center">
  <img src="assert/last.png" width="640" alt="WePM architecture">
</p>

## Performance Across Three Capability Families

WePM is evaluated across all three capability families rather than through a single aggregate score: **Embedding** covers 20 datasets and 10 models, **Generation** covers 36 datasets and 9 models, and **Decision** covers 4 datasets and 17 models. The figure summarizes the benchmark-level comparisons and reports WePM's overall rank within each evaluation pool.

<p align="center">
  <img src="assert/performance.png" width="1080" alt="Per-benchmark ranking of WePM across the Embedding, Generation and Decision capability families">
</p>

WePM ranks first overall in all three families. The result is important for a unified model: adding generation and typed decision paths does not come at the expense of multimodal representation, while strengthening representation does not dilute OCR, visual reasoning or decision quality. The model therefore expands the range of supported tasks without reducing any capability family to a nominal add-on.

## Heterogeneous Capability Density (HCD)

Benchmark quality alone does not capture the central constraint of on-device deployment: **how much usable intelligence can remain resident within a fixed memory budget?** We introduce **Heterogeneous Capability Density (HCD)** to jointly account for capability breadth, capability quality and resident inference memory.

<p align="center">
  <img src="assert/hcd_formula.png" width="640" alt="HCD = (number of capability families above the usability threshold) x (aggregate capability quality) / (resident inference memory)">
</p>

HCD counts capability families rather than individual benchmarks. For example, OCR, captioning and visual question answering all contribute to Generation instead of being counted as three separate capabilities. A higher HCD therefore indicates that a model delivers broader and stronger usable intelligence per unit of resident memory—not merely that it has been evaluated on more tasks.

<p align="center">
  <img src="assert/memory_vs_quality.png" width="640" alt="HCD comparison between WePM and six independent three-model compositions">
</p>

The comparison pairs independent generation and embedding models with the same 4.5B decision model. WePM reaches approximately 68% harmonic performance with about 8 GB of BF16 weight memory, yielding an HCD of 0.255. The six independent three-model compositions require roughly 21–41 GB, yet achieve HCD values of only 0.049–0.096. In this comparison, unifying the three capability families provides substantially higher capability density than stacking specialized models.

The memory values above count BF16 weights only. KV cache, activations, framework overhead and runtime fragmentation are excluded.

## Applications

WePM is intended as a shared local intelligence layer rather than a collection of benchmark-specific interfaces. The following demos run fully on device, keeping personal images, text and interaction data local.

### 1. Personal Agent

A single resident WePM-4B supports the complete local workflow: it retrieves relevant personal content, understands multimodal context, generates a grounded response and makes the typed decision that routes the next step. Retrieval, generation and decision are composed within one model instead of being dispatched to separate services.

<p align="center">
  <video src="assert/personal_agent_demo.mp4" poster="assert/personal_agent_poster.jpg" width="640" controls muted loop playsinline></video>
</p>

If the player above is not rendered by your Markdown viewer, open the [MP4 demo](assert/personal_agent_demo.mp4) directly.

<!-- TBD: capture device, memory footprint, precision and latency numbers for this demo. -->

### 2. On-Device Game Assistant

The same WePM-4B interprets game frames and interface text, retrieves local knowledge and interaction history, generates contextual suggestions, and performs option judgment or action routing. This turns one continuously resident model into a complete multimodal game assistant without separately loading vision, retrieval and decision models.

<p align="center">
  <video src="assert/game_assistant_demo.mp4" poster="assert/game_assistant_poster.jpg" width="640" controls muted loop playsinline></video>
</p>

If the player above is not rendered by your Markdown viewer, open the [MP4 demo](assert/game_assistant_demo.mp4) directly.

## Inference

The repository keeps runtime code, runnable cases, and model weights separate:

```text
inference/              # Complete inference implementation
examples/gpu/           # NVIDIA GPU task cases and input assets
examples/cpu/           # CPU task cases and input assets
checkpoints/huggingface # Hugging Face safetensors checkpoint
checkpoints/gguf/       # Hybrid GGUF bundle (not ONNX)
```

The current runnable examples use the complete Hugging Face checkpoint and support generation, embedding extraction, and structured decision. Generation cases cover single-text, single-image, and ordered text–image interleaving.

### Hardware and precision

| Device | Compute | Approximate weight memory | Recommended memory |
| --- | --- | --- | --- |
| NVIDIA GPU | BF16 | ≈8 GB | ≥16 GB GPU memory |
| CPU | FP32 | ≈17 GiB | ≥24 GiB system memory |

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
```

NVIDIA GPU (CUDA 12.8 example):

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r inference/requirements.txt
```

CPU:

```bash
python -m pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r inference/requirements.txt
```

The scripts use `checkpoints/huggingface` by default. Set `WEPM_MODEL` to use another complete Hugging Face checkpoint.

### NVIDIA GPU inference

#### Generation: single text, single image, or interleaved content

```bash
bash examples/gpu/generation/run.sh single_text
bash examples/gpu/generation/run.sh single_image
bash examples/gpu/generation/run.sh interleaved
```

#### Embedding extraction

```bash
bash examples/gpu/embedding/run.sh
```

#### Structured decision

```bash
bash examples/gpu/decision/run.sh
```

### CPU inference

#### Generation: single text, single image, or interleaved content

```bash
bash examples/cpu/generation/run.sh single_text
bash examples/cpu/generation/run.sh single_image
bash examples/cpu/generation/run.sh interleaved
```

#### Embedding extraction

```bash
bash examples/cpu/embedding/run.sh
```

#### Structured decision

```bash
bash examples/cpu/decision/run.sh
```

Each command writes a timestamped JSON result, releases the model, and exits. See the [inference guide](inference/README.md) for the CLI and ordered multimodal input format, and the [examples guide](examples/README.md) for all runnable cases.

## Citation

If you find this work useful, please consider citing it:

```bibtex
@misc{wepm2026,
  title = {WePM: Hierarchical Conflict Reconciliation for Unified MLLMs},
  author = {{WeChat AI, Tencent}},
  year = {2026},
  howpublished = {Technical Report},
  url = {TBD}
}
```

## License

TBD. Third-party components remain subject to their original licenses.
