# Qwen4Exp PLE rows on SSD

Qwen3.8-Flash-Next carries a 320M-row n-gram (PLE) embedding table: 47.7 GiB in the FP8
checkpoint. With `--engram-config '{"cpu_offload": true}'` vLLM keeps it in pinned host memory.
This tool and the `ssd_rows_path` option serve it from local storage instead, so the host memory
is only used as reclaimable page cache.

## 1. Export the table once

```bash
python3 tools/ple_ssd/export_ple_rows.py --model /path/to/Qwen3.8-Flash-Next-NVFP4 --output /data/ple-rows
```

The output directory holds `ple_rows.fp8.bin`, with FP8 rows of 160 bytes packed 25 per 4 KiB
page so that every row is one aligned read, and `ple_rows.json`, which holds the layout, scale,
SHA-256 and a sampled byte-for-byte verification against the checkpoint. The export streams, needs
~100 MiB of RAM and ~52.4 GB of disk.

## 2. Serve

```bash
vllm serve /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --engram-config '{"cpu_offload": true, "ssd_rows_path": "/data/ple-rows/ple_rows.json"}' ...
```

Checkpoint PLE rows are then skipped during loading, and nothing is pinned for the table. Each step
reads its rows with parallel `pread` calls, using a small C++ extension built on first use. Serving
requires the extension; only the tensor-level reader used in tests has a Python fallback. The
table's global scale is checked against the checkpoint.

Each lookup is stream-ordered: a device-to-host copy of the row IDs, a CUDA host function
(`cudaLaunchHostFunc`) that runs the `pread` calls, and a host-to-device copy of the rows. The
lookup therefore also works inside captured CUDA graphs.

To use it with an existing vLLM image, build `tools/ple_ssd/Dockerfile`.

## Measured on IGX Thor + RTX PRO 6000 (23 September 2026)

The comparison is against the pinned-host table, with the same flags: CUDA graphs, MTP 2, prefix
caching, 8K chunks, 10 GiB FP8 KV. The row file was evicted from the page cache before the run.

| | pinned table | SSD rows |
|---|---:|---:|
| Server start to ready | 362 s | 231 s |
| Pinned host memory for the table | 47.7 GiB | 0 |
| Host memory available while serving | ~51 GiB | ≥ 114.6 GiB |
| First token, 8K / 128K / 256K prompt | 0.37 / 8.45 / 11.96 s | 0.39 / 8.62 / 12.15 s |
| First token, first cold 8K natural-text prompt | 0.68 s | 1.56 s |
| Decode, 1 client, 8K / 256K | 166 / 186 tok/s | 151 / 186 tok/s |
| Output, 32K × 8 clients | 425 tok/s | 405 tok/s |

Generated text is byte-identical to the pinned table for all 13 measured requests that loaded
their prompt from scratch: 5 single-client prompts plus 12 concurrent streams. It also matches the
pre-optimization eager baseline for the fixed 8K prompt. One repeated prompt that was served from
the prefix cache differed from the pinned run after 92 characters. On the pinned table, that same
cached repeat also differs from its own first run; on SSD rows it matches. The difference therefore
comes from the prefix-cache path, not from the row source.

Limits: FP8 PLE checkpoints only, one embedding-parallel rank (`ETP=1`). Rows are fetched at
the start of each step on the PLE side stream. Cold prefill therefore depends on SSD random-read
throughput (about 16 reads per prompt token, minus page-cache hits). Rows are not yet prefetched
when a request is admitted.
