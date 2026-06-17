# Profile Guide

Language: English | [简体中文](README.zh-CN.md)

This directory contains launch profiles for vLLM Tesla T10 Definitive. A profile
is an `.env` preset for runtime parameters only; it does not include the
checkpoint path. Choose the model directory separately in `launcher.sh` or with
`MODEL_DIR=...`.

Tesla T10 routes are organized by tensor-parallel size under `tp<N>/`. Set
`GPU_DEVICES` and `TP_SIZE` to the same GPU count when launching (for example
TP=4 with `GPU_DEVICES=0,1,2,3`).

## Tesla T10 profiles (experimental)

Starting presets for Tesla T10 16GB hosts. Throughput numbers are **not**
validated yet; run the sweep in
[Qwen3.6 KV Throughput Sweep (Tesla T10)](../docs/qwen36-kv-throughput-sweep-t10.md)
before promoting any route.

Tested checkpoint target: GPTQ-INT4, about 19G.

| Profile | TP | Compatible modes | Context | KV | MTP | Status |
|---|---|---:|---|---:|---:|---|
| `qwen27b/tp2/normal/int4/int8kv-96K-mtp3-text-only.env` | 2 | normal | 96K | INT8 | 3 | experimental |
| `qwen27b/tp2/fast/int4/tqk8v4-128K-mtp3-text-only.env` | 2 | fast | 128K | TQK8V4 | 3 | experimental |
| `qwen27b/tp4/normal/int4/fp16kv-128K-mtp3-text-only.env` | 4 | normal | 128K | FP16 | 3 | experimental |
| `qwen27b/tp4/normal/int4/int8kv-256K-mtp3-text-only.env` | 4 | normal | 256K | INT8 | 3 | experimental |

Profile layout:

```text
profiles/
  templates/
  qwen27b/
    tp2/
      normal/int4/
      fast/int4/
    tp4/
      normal/int4/
    normal/          # legacy 2080 Ti validated routes
      fp8/
      int4/
    fast/
      fp8/
      int4/
    user/
```

Launch modes:

- `safe`: conservative fallback mode for maximum compatibility.
- `normal`: recommended daily mode for stable deployments.
- `fast`: high-performance mode for higher throughput.
- `aggressive`: more aggressive mode with the highest performance and quality risk.

`profiles/templates/` contains optional chat-template presets. They are selected
from the launcher as a global service setting; route profiles do not store chat
templates, GPU devices, ports, reasoning defaults, or tool-calling defaults.

File names describe the intended route:

```text
<kv-precision>-<context>-<mtp>-<message-type>.env
```

KV positioning:

- `fp16kv`: quality route.
- `int8kv`: capacity / balance route.
- `tqk8v4`: TurboQuant K8V4 compression route for `fast` profiles.

## Legacy validated profiles (dual RTX 2080 Ti 22GB, TP=2)

The tables below remain as historical evidence from the original 2080 Ti fork.
Do **not** deploy them on Tesla T10 without re-validation.

### FP8

Tested checkpoint: Jackrong/Qwopus3.6-27B-v2-FP8, about 29G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/fp8/fp16kv-128K-mtp3-text-only.env` | normal | 128K | FP16 | 3 | text-only | 1 | 1619.48 / 84.71 |
| `qwen27b/normal/fp8/int8kv-252K-mtp3-text-only.env` | normal | 252K | INT8 | 3 | text-only | 1 | 1605.10 / 44.09 |
| `qwen27b/fast/fp8/fp16kv-112K-mtp3-text-only.env` | fast | 112K | FP16 | 3 | text-only | 1 | 1615.58 / 83.69 |
| `qwen27b/fast/fp8/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1615.81 / 81.06 |
| `qwen27b/fast/fp8/tqk8v4-240K-mtp3-text-image.env` | fast | 240K | TQK8V4 | 3 | text+image | 1 | 1605.61 / 80.67 |

### AWQ/GPTQ-INT4

Tested checkpoint: GPTQ-INT4, about 19G.

| Profile | Compatible modes | Context | KV | MTP | Messages | Seqs | Throughput |
|---|---|---:|---|---:|---|---:|---:|
| `qwen27b/normal/int4/fp16kv-256K-mtp3-text-only.env` | normal | 256K | FP16 | 3 | text-only | 1 | 1738.06 / 97.79 |
| `qwen27b/normal/int4/fp16kv-240K-mtp3-text-image.env` | normal | 240K | FP16 | 3 | text+image | 1 | 1760.14 / 94.48 |
| `qwen27b/normal/int4/int8kv-two250K-mtp3-text-only.env` | normal | 250K per workspace | INT8 | 3 | text-only | 2 | 1740.51 / 49.06 |
| `qwen27b/normal/int4/int8kv-512K-yarn-mtp3-text-only.env` | normal | 512K | INT8 + YaRN | 3 | text-only | 1 | 1734.14 / 48.16 |
| `qwen27b/fast/int4/fp16kv-256K-mtp3-text-only.env` | fast | 256K | FP16 | 3 | text-only | 1 | 1734.98 / 87.00 |
| `qwen27b/fast/int4/tqk8v4-256K-mtp3-text-only.env` | fast | 256K | TQK8V4 | 3 | text-only | 1 | 1744.67 / 100.81 |
| `qwen27b/fast/int4/tqk8v4-two250K-mtp3-text-only.env` | fast | 250K per workspace | TQK8V4 | 3 | text-only | 2 | 1739.23 / 99.91 |

Throughput uses the `4096/128` test shape and is shown as
`prefill tok/s / decode tok/s`. Chinese quality smoke is run before throughput;
routes that fail quality are not kept as profiles.
