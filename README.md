<!-- markdownlint-disable MD001 MD041 -->
# ⚡ vLLM Tesla T10 Definitive Edition

![vLLM Tesla T10 Definitive Edition cover](docs/assets/vllm-2080ti-cover.jpg)

The definitive vLLM runtime for multi-GPU Tesla T10 / SM75 serving.

This is a hardware-focused fork that preserves the patched source, launch
profiles, and runtime notes needed to reproduce the working Tesla T10 vLLM stack
on SM75 Turing GPUs.

Fork release: `v0.2.0`
Base vLLM: `0.21.0`

Headline target: Qwen3.6 27B on multi Tesla T10 16GB with flexible tensor
parallel size (TP=2/4/8). Throughput numbers require re-validation on your
exact GPU count and PCIe topology.

Language: English | [简体中文](README.zh-CN.md)

![Live single-request throughput demo](docs/assets/vllmspeed.gif)

## 💡 Why Tesla T10 for LLM Inference?

Tesla T10 (TU102, SM75) is a datacenter Turing GPU with 16GB GDDR6 and about
403 GB/s memory bandwidth. It shares the same compute capability as RTX 2080 Ti
and other Turing cards, so this fork reuses the validated SM75 kernel stack
(Marlin, FlashQLA, FlashInfer, TurboQuant/INT8 KV, MTP, CUDAGraph).

| Metric | Tesla T10 16GB | RTX 2080 Ti 22GB (legacy target) | Notes |
|---|---:|---:|---|
| Architecture | TU102 / SM75 | TU102 / SM75 | Same CUDA arch (`7.5`) |
| VRAM | 16GB | 22GB (modded) | T10 profiles use tighter `GPU_UTIL` |
| Memory bandwidth | 403 GB/s | 616 GB/s | Expect lower throughput |
| Tensor cores | 448 | 544 | Slightly lower compute |
| Interconnect | PCIe P2P typical | NVLink common | Multi-card TP relies on P2P |
| TDP | 150W | 250W | Easier to pack multiple cards |

The project optimizes for **multiple Tesla T10 cards** with configurable tensor
parallel size. TP=2 fits 27B INT4/INT8KV routes on 32GB aggregate VRAM; TP=4
and TP=8 unlock longer context at the cost of PCIe all-reduce overhead.

That is the value of this fork: take proven SM75 runtime work and adapt launch
detection, profiles, and documentation for Tesla T10 multi-GPU hosts.

## 🧩 Core Routes

Serving shape:

- This project optimizes for extreme single-concurrency performance on multi
  Tesla T10: one personal-agent style workload, one serious 27B/31B model, and
  the largest practical context window this hardware can sustain at the chosen TP
  size.
- It is not a multi-tenant serving stack. Multi-agent use is supported best as
  queued workspace isolation, not as parallel long-prefill throughput. Long
  prefill work is capacity-safe when tuned, but it is effectively serialized by
  but in this multi-T10 profile set it is effectively serialized by the runtime
  scheduler when TP spans multiple GPUs.

Status: 🟢 validated support; 🟡 experimental or partial support; 🔴 known
failure or clear regression; ⚪ not a target preset or not yet validated.

### Qwen3.6 27B Mature Route

Qwen-family 27B is the primary production route for this fork. It has the
broadest tested coverage across FP8/INT4/NVFP4 weights, MTP, FP16/INT8/
TurboQuant KV, 256K native context, YaRN capacity, and image serving.

| Feature | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin weight route | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 | 🟢 FP8/INT4/NVFP4 |
| MTP decoding | 🟢 supported | 🟢 supported | 🟢 supported |
| Native 256K context | 🟢 supported | 🟢 supported | 🟢 supported |
| YaRN extension | ⚪ not the target route | 🟢 supported | ⚪ not a target preset |
| No-eager / CUDAGraph | 🟢 supported | 🟡 partial support | 🟢 supported |
| Fast prefill path | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer | 🟢 FlashQLA / FlashInfer |
| Multimodal image serving | 🟢 supported | 🟢 supported | 🟢 supported |
| Current preset status | 🟢 normal / fast / safe | 🟢 normal / safe | 🟢 fast |

### Gemma4 31B Experimental Route

Gemma4 31B is kept as a secondary experimental route. The official QAT target
with the matching QAT assistant is now the most promising Gemma path, with
better FP16/default-KV headroom than earlier Gemma checkpoints.

| Feature | FP16 KV | INT8 KV | TurboQuant KV |
|---|---|---|---|
| Marlin weight route | 🟢 GPTQ / QAT | 🟡 GPTQ / QAT | 🟡 GPTQ / QAT |
| MTP decoding | 🟡 QAT assistant MTP3 | ⚪ no preset | ⚪ no preset |
| Validated context | 🟡 about 170K KV headroom observed | 🔴 init issue | 🔴 capacity shortfall |
| No-eager / CUDAGraph | 🟢 supported | 🟡 fallback issue | 🟡 admission limited |
| Fast prefill path | 🟢 FlashInfer | 🟡 FlashInfer | 🟡 FlashInfer |
| Multimodal image serving | ⚪ no validated preset | ⚪ no validated preset | ⚪ no validated preset |
| Current preset status | 🟡 experimental only | ⚪ no preset | ⚪ no preset |

## 🧪 Tested Model Checkpoints

This section records checkpoint-level validation. It is intentionally stricter
than "vLLM can load it": a supported checkpoint can start and generate, while a
recommended checkpoint also has a useful throughput/context tradeoff on Tesla T10
at the documented TP size.

| Model route | Weight route | Model cards | Status |
|---|---|---|---|
| Qwen3.6 27B FP8 | FP8 | [Qwen/Qwen3.6-27B-FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8)<br>[Jackrong/Qwopus3.6-27B-v2-FP8](https://huggingface.co/Jackrong/Qwopus3.6-27B-v2-FP8) | 🟢 Recommended |
| Qwen3.6 27B AWQ | AWQ-INT4 | [QuantTrio/Qwen3.6-27B-AWQ](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ)<br>[mconcat/Qwopus3.6-27B-v2-AWQ-4bit](https://huggingface.co/mconcat/Qwopus3.6-27B-v2-AWQ-4bit) | 🟢 Recommended |
| Qwen3.6 27B GPTQ | GPTQ-INT4 | [llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GPTQ-Int4) | 🟢 Recommended |
| Qwen3.6 27B NVFP4 | NVFP4 | [unsloth/Qwen3.6-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4) | 🟡 Supported |
| Qwen3.6 27B Quark INT8 | Quark-INT8 | [nameistoken/Qwen3.6-27B-Quark-W8A8-INT8](https://huggingface.co/nameistoken/Qwen3.6-27B-Quark-W8A8-INT8) | 🟡 Supported |
| Qwen3.6 27B AutoRound | AutoGPTQ-INT8 | [Minachist/Qwen3.6-27B-INT8-AutoRound](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound)<br>[Minachist/Qwen3.6-27B-INT8-AutoRound W8A16-GS128](https://huggingface.co/Minachist/Qwen3.6-27B-INT8-AutoRound/tree/W8A16-GS128) | 🟡 Supported |
| Gemma4 31B QAT | QAT + QAT assistant draft | [google/gemma-4-31B-it-qat-w4a16-ct](https://huggingface.co/google/gemma-4-31B-it-qat-w4a16-ct)<br>[google/gemma-4-31B-it-qat-q4_0-unquantized-assistant](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized-assistant) | 🟡 Supported |
| Gemma4 31B GPTQ | GPTQ-INT4 + assistant draft | [ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ](https://huggingface.co/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ) | 🟡 Supported |

## 🛠️ Target Hardware & Runtime

- Validated GPU profile: multi Tesla T10 16GB, SM75, PCIe P2P, tensor parallel
  size 2/4/8 (flexible; TP must match selected GPU count)
- Validated host OS: Ubuntu 22.04/24.04 LTS or Debian 12, on Linux kernel 6.x
- CUDA/PyTorch: CUDA 12.8, `torch 2.11.0+cu128`
- Fork release: `v0.2.0`
- Base vLLM: `0.21.0`
- Repository identity: `vllm-tesla-t10-definitive`
- Runtime identity: `vllm-sm75-tesla-t10-cu128`
- Compatibility target: NVIDIA Turing / SM75 GPUs. Tesla T4 is build-compatible
  but not the primary profile target. Other Turing cards still need profile
  validation for VRAM capacity, P2P/NVLink behavior, model head_dim, KV dtype,
  and CUDAGraph/MTP settings.

## 🚀 How To Use

For a source checkout, use two steps.

1. Build the runtime:

```bash
./build.sh
```

`build.sh` creates the local `.venv`, installs dependencies, builds the CUDA
extensions, and prints a clear success or failure result with a build log path.

2. Start and manage the service:

```bash
./launcher.sh
```

The launcher is the interactive service manager. From the menu you can choose
the checkpoint directory, apply or edit a profile, select `safe` / `normal` /
`fast`, choose GPU/TP devices, set the port, switch local-only or LAN access,
configure chat templates and tool calling, start the server, stop the server,
or save a custom profile.

Tool calling is supported through the OpenAI-compatible serving API. The
launcher exposes automatic tool choice, tool parser selection, and strict
structured tool output as global runtime settings.

After a successful launch, the status panel shows `RUNNING`, the served model
name, PID, OpenAI-compatible API URL, log file, and cache capacity when vLLM
reports it.

For scripted use:

```bash
MODEL_DIR=/path/to/qwen-or-gemma-checkpoint \
PROFILE=qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env \
MODE=normal \
PORT=8000 \
SERVICE_SCOPE=lan \
GPU_DEVICES=0,1,2,3 \
TP_SIZE=4 \
./launcher.sh --non-interactive
```

Profiles declare compatible modes, not a recommended launch mode. Set
`MODE=safe`, `MODE=normal`, or `MODE=fast` explicitly when you want a specific
mode; the launcher validates that choice against the profile.

3. Update an existing checkout:

```bash
./update.sh
```

`update.sh` checks the latest GitHub Release against the local fork version,
downloads the newer release archive when available, preserves local runtime
state such as `.venv`, `.deps`, logs, results, caches, and user profiles, then
asks whether to run `build.sh` immediately.

## 🧭 Profiles

Start from [Profile Guide](profiles/README.md). Tesla T10 routes live under
`profiles/<model>/tp<N>/<mode>/<weight>/<route>.env`, for example
`qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env` and
`qwen27b/tp4/normal/int4/fp16kv-128K-mtp3-text-only.env`.

Legacy 2080 Ti validated routes remain under `profiles/qwen27b/normal/...` for
reference but are not promoted for Tesla T10 deployment.

Available modes:

- `normal`: recommended default mode for regular production deployments.
- `fast`: high-performance mode, not recommended for stable production
  deployments.
- `aggressive`: more aggressive mode with the highest performance and quality risk.
- `safe`: safety mode. It is slower, but prioritizes highly stable output
  quality for troubleshooting.

## 🚀 MTP And KV Precision

Use the bundled profiles instead of hand-tuning MTP and KV settings first.
MTP is already set to the best practical value for each route. Choose KV by
intent first: FP16/default KV for maximum output quality, INT8 KV for balanced
long-context service, and TurboQuant K8V4 for fast compression routes.

Detailed benchmark notes are kept in
[MTP Task Sensitivity](docs/mtp-task-sensitivity.md) and
[Qwen3.6 KV Throughput Sweep](docs/qwen36-kv-throughput-sweep.md).

## ❓ Hardware Q&A

**Q: What GPU interconnect is required?**

A: PCIe peer-to-peer (P2P) is the baseline for multi Tesla T10 tensor parallel.
NVLink is not available on T10. Confirm topology with `nvidia-smi topo -m`
before promoting a TP=4/8 route. Without working P2P, multi-GPU TP may fall back
to slower paths or fail to start.

**Q: Does the host need a strong CPU or a lot of RAM?**

A: It does not need a high-end CPU, but it does prefer a modern CPU with strong
single-core performance and low platform latency. More RAM mainly helps builds,
downloads, and compile cache. Because vLLM has a Python/service control plane,
very old CPUs may be better matched to minimal C++ runtimes such as llama.cpp.

**Q: Which Turing GPUs make sense? What about Tesla T4?**

A: The primary validated target is **multi Tesla T10 16GB** with TP matching
GPU count. Tesla T4 is SM75-compatible and can run smaller models or short
context routes, but it is not the main profile target for 27B/31B serving.
Legacy dual RTX 2080 Ti 22GB routes remain documented under
`profiles/qwen27b/normal/...` for reference.

**Q: How do I select GPUs and TP size?**

A: Set `TARGET_GPU_PATTERN=t10` (default) so the launcher auto-selects all Tesla
T10 cards, or set `GPU_DEVICES` explicitly. `TP_SIZE` must equal the number of
selected GPUs. Use menu item 3 or pass both variables in non-interactive mode.

**Q: Which CUDA, PyTorch, and driver versions are validated?**

A: The validated runtime is CUDA 12.8 + `torch 2.11.0+cu128`. The reference
validation host used NVIDIA driver `590.48.01`. Use a recent NVIDIA driver that
supports your host GPUs and is compatible with the CUDA runtime. Do not mix
build/runtime assumptions casually: keep the PyTorch CUDA lane, local CUDA
toolkit, FlashInfer/FlashQLA builds, and launch profile aligned.

**Q: What other hardware risks matter?**

A: Cooling, power stability, and enough SSD space for model files and compile
caches. Thermal throttling can hide as a software regression, especially during
long prefill or repeated CUDAGraph/AOT compilation runs.

## 🔗 Related Project

- [2080Ti-LLM-Toolbox](https://github.com/weicj/2080Ti-LLM-Toolbox): companion
  toolbox for dual 2080 Ti model routes, benchmark summaries, model notes, and
  operational guidance. This repository focuses on the patched vLLM runtime
  itself.

## 🙏 Credits / Upstream Projects

This repository is a hardware-focused fork of upstream
[vLLM](https://github.com/vllm-project/vllm), licensed under Apache-2.0. The
fork keeps the upstream project structure and adds local SM75 runtime patches,
launch profiles, and validation notes for the Tesla T10 multi-GPU route.

Acceleration components used or integrated by this runtime include:

- [vLLM](https://github.com/vllm-project/vllm): base inference engine and
  serving stack.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): attention,
  sampling, and quantized kernel paths used by vLLM.
- [QwenLM/FlashQLA](https://github.com/QwenLM/FlashQLA): upstream FlashQLA
  Gated DeltaNet / Qwen3.5 linear-attention implementation.
- [weicj/FlashQLA-SM70-SM75](https://github.com/weicj/FlashQLA-SM70-SM75):
  SM70/SM75 adaptation used by the validated Qwen3.6 prefill profile.
- TurboQuant, Marlin, CUTLASS, Triton, and related vLLM
  acceleration kernels: existing open-source acceleration work integrated and
  profiled for this hardware target.

While this Tesla T10 fork won't strictly follow the main fork vLLM,
the patches merged by vLLM update will be re-validated under SM75 specific scope.

