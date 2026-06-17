# Qwen3.6 27B KV / Throughput Sweep (Tesla T10)

Language: English | [简体中文](qwen36-kv-throughput-sweep-t10.zh-CN.md)

This document describes how to re-validate Qwen3.6 27B routes on Tesla T10 16GB
hosts. It follows the same methodology as
[Qwen3.6 KV Throughput Sweep](qwen36-kv-throughput-sweep.md), but the capacity
assumptions differ from the legacy dual RTX 2080 Ti 22GB target.

## Hardware assumptions

- GPU: Tesla T10 16GB, SM75 (TU102)
- Interconnect: PCIe P2P required for TP > 1
- Tensor parallel: TP must equal selected GPU count (`GPU_DEVICES` length)
- Starting profiles: see `profiles/qwen27b/tp2/...` and `profiles/qwen27b/tp4/...`

## Capacity notes (pre-benchmark)

| TP | Aggregate VRAM | Starting route | Status |
|---:|---:|---|---|
| 2 | 32GB | INT8KV 96K, TQK8V4 128K | experimental |
| 4 | 64GB | FP16KV 128K, INT8KV 256K | experimental |
| 8 | 128GB | not shipped yet | not promoted |

27B GPTQ-INT4 weights are about 19GB total. TP=2 leaves little headroom for
long FP16 KV routes on 16GB cards; prefer INT8KV or TQK8V4 until benchmark
evidence is collected.

## Sweep procedure

1. Confirm topology: `nvidia-smi topo -m`
2. Build runtime: `./build.sh`
3. Pick a starting profile under `profiles/qwen27b/tp<N>/...`
4. Launch with explicit GPU/TP settings:

```bash
MODEL_DIR=/path/to/gptq-int4-checkpoint \
PROFILE=qwen27b/tp4/normal/int4/int8kv-256K-mtp3-text-only.env \
MODE=normal \
GPU_DEVICES=0,1,2,3 \
TP_SIZE=4 \
./launcher.sh --print-config
```

5. Run capacity smoke (OOM boundary), then throughput at `4096/128`
6. Record prefill/decode tok/s and Chinese quality smoke result
7. Promote only routes that pass both capacity and quality checks

## Promotion rules

- Routes without benchmark numbers stay `experimental`
- Do not copy 2080 Ti throughput tables directly; PCIe bandwidth and 16GB VRAM
  change both capacity and decode speed
- Update `profiles/README.md` when a route is promoted to `validated`
