# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp PLE rows served from a page-aligned row file on local storage.

The pinned-host backend keeps the whole n-gram table (47.7 GiB for the FP8
Qwen3.8-Flash-Next checkpoint) in pinned memory. This backend keeps only a
row file on disk, written by ``tools/ple_ssd/export_ple_rows.py``, and reads
the rows each step needs with parallel ``pread`` calls. The OS page cache holds
recently used pages, so frequently used n-grams stay in memory while the
kernel can still reclaim them under pressure.

File layout: row ``r`` is at ``(r // rows_per_page) * page_bytes +
(r % rows_per_page) * row_bytes``; a row never crosses a page boundary.
"""

import json
import os
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.logger import init_logger

from .ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEFp8EmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
)

logger = init_logger(__name__)

ROW_FILE_FORMAT = "qwen4-exp-ple-rows"

_READER_SOURCE = r"""
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

// Copy table rows id[i] into dst[i] with parallel pread. Rows outside
// [0, num_rows) are zero-filled, matching the pinned backend's masking.
static int64_t gather_core(int fd, const int64_t* id, uint8_t* dst, int64_t n,
                           int64_t row_bytes, int64_t rows_per_page,
                           int64_t page_bytes, int64_t num_rows,
                           int64_t max_threads) {
  std::atomic<int64_t> errors{0};
  auto work = [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      const int64_t row = id[i];
      uint8_t* d = dst + i * row_bytes;
      if (row < 0 || row >= num_rows) {
        std::memset(d, 0, row_bytes);
        continue;
      }
      const off_t offset = static_cast<off_t>(row / rows_per_page) * page_bytes +
                           static_cast<off_t>(row % rows_per_page) * row_bytes;
      int64_t done = 0;
      while (done < row_bytes) {
        const ssize_t got = pread(fd, d + done, row_bytes - done, offset + done);
        if (got <= 0) {
          errors.fetch_add(1);
          std::memset(d, 0, row_bytes);
          break;
        }
        done += got;
      }
    }
  };
  // Small (decode) batches stay on the calling thread; large (prefill) batches
  // fan out so cold rows are read at a useful queue depth.
  const int64_t threads = std::max<int64_t>(1, std::min<int64_t>(max_threads, n / 256));
  if (threads == 1) {
    work(0, n);
  } else {
    std::vector<std::thread> pool;
    const int64_t chunk = (n + threads - 1) / threads;
    for (int64_t t = 0; t < threads; ++t) {
      const int64_t begin = t * chunk, end = std::min(n, begin + chunk);
      if (begin < end) pool.emplace_back(work, begin, end);
    }
    for (auto& th : pool) th.join();
  }
  return errors.load();
}

int64_t gather_rows(int64_t fd, torch::Tensor ids, torch::Tensor out,
                    int64_t row_bytes, int64_t rows_per_page, int64_t page_bytes,
                    int64_t num_rows, int64_t max_threads) {
  TORCH_CHECK(ids.device().is_cpu() && ids.scalar_type() == torch::kInt64 &&
              ids.is_contiguous(), "ids must be a contiguous CPU int64 tensor");
  TORCH_CHECK(out.device().is_cpu() && out.scalar_type() == torch::kUInt8 &&
              out.is_contiguous(), "out must be a contiguous CPU uint8 tensor");
  const int64_t n = ids.numel();
  TORCH_CHECK(out.numel() >= n * row_bytes, "out is too small");
  return gather_core(static_cast<int>(fd), ids.data_ptr<int64_t>(),
                     out.data_ptr<uint8_t>(), n, row_bytes, rows_per_page,
                     page_bytes, num_rows, max_threads);
}

// Stream-ordered gather for CUDA graphs: a host node reads the row IDs that a
// preceding device-to-host copy placed in pinned memory and fills pinned rows
// for a following host-to-device copy. Parameters live until process exit
// because captured graphs keep pointing at them.
struct GatherParams {
  int fd;
  const int64_t* ids;
  uint8_t* rows;
  int64_t n, row_bytes, rows_per_page, page_bytes, num_rows, max_threads;
};
static std::atomic<int64_t> g_errors{0};

static void CUDART_CB gather_host_fn(void* data) {
  const auto* p = static_cast<const GatherParams*>(data);
  g_errors.fetch_add(gather_core(p->fd, p->ids, p->rows, p->n, p->row_bytes,
                                 p->rows_per_page, p->page_bytes, p->num_rows,
                                 p->max_threads));
}

int64_t make_params(int64_t fd, torch::Tensor ids, torch::Tensor rows, int64_t n,
                    int64_t row_bytes, int64_t rows_per_page, int64_t page_bytes,
                    int64_t num_rows, int64_t max_threads) {
  TORCH_CHECK(ids.is_pinned() && rows.is_pinned(), "staging buffers must be pinned");
  TORCH_CHECK(ids.numel() >= n && rows.numel() >= n * row_bytes, "staging too small");
  auto* p = new GatherParams{static_cast<int>(fd), ids.data_ptr<int64_t>(),
                             rows.data_ptr<uint8_t>(), n, row_bytes, rows_per_page,
                             page_bytes, num_rows, max_threads};
  return reinterpret_cast<int64_t>(p);
}

void enqueue_gather(int64_t stream, int64_t params) {
  const cudaError_t err = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream),
                                             gather_host_fn,
                                             reinterpret_cast<void*>(params));
  TORCH_CHECK(err == cudaSuccess, "cudaLaunchHostFunc failed: ", cudaGetErrorString(err));
}

int64_t take_errors() { return g_errors.exchange(0); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_rows", &gather_rows, pybind11::call_guard<pybind11::gil_scoped_release>());
  m.def("make_params", &make_params);
  m.def("enqueue_gather", &enqueue_gather);
  m.def("take_errors", &take_errors);
}
"""

RowReader = Callable[..., int]
_module = None


def _python_gather_rows(fd, ids, out, row_bytes, rows_per_page, page_bytes,
                        num_rows, max_threads) -> int:
    """Slow reference implementation used when the extension cannot build."""
    del max_threads
    errors = 0
    rows = out.view(-1, row_bytes)
    for i, row in enumerate(ids.tolist()):
        if not 0 <= row < num_rows:
            rows[i].zero_()
            continue
        offset = (row // rows_per_page) * page_bytes + (row % rows_per_page) * row_bytes
        data = os.pread(fd, row_bytes, offset)
        if len(data) != row_bytes:
            errors += 1
            rows[i].zero_()
            continue
        rows[i].copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8))
    return errors


def load_reader_module():
    """Build (once) and return the C++ reader extension."""
    global _module
    if _module is None:
        from torch.utils.cpp_extension import load_inline

        _module = load_inline(
            name="qwen4_exp_ple_ssd_reader_v2",
            cpp_sources=[_READER_SOURCE],
            extra_cflags=["-O3"],
            with_cuda=True,
            verbose=False,
        )
    return _module


def load_row_reader(allow_fallback: bool = True) -> RowReader:
    """Return the tensor-level parallel reader (the Python one if the build fails)."""
    try:
        return load_reader_module().gather_rows
    except Exception as exc:  # noqa: BLE001 - any build failure falls back
        if not allow_fallback:
            raise
        logger.warning(
            "Could not build the SSD PLE row reader (%s); using the slow Python "
            "reader instead.", exc)
        return _python_gather_rows


def load_row_file_metadata(path: str) -> dict:
    """Read and validate the metadata JSON written by the row-file exporter."""
    meta_path = Path(path)
    meta = json.loads(meta_path.read_text())
    if meta.get("format") != ROW_FILE_FORMAT or meta.get("version") != 1:
        raise ValueError(f"{path} is not a version-1 {ROW_FILE_FORMAT} file")
    if meta.get("dtype") != "float8_e4m3fn":
        raise ValueError(f"Unsupported PLE row dtype {meta.get('dtype')!r}")
    data = meta_path.parent / meta["file"]
    rows, row_bytes = int(meta["rows"]), int(meta["row_bytes"])
    rows_per_page, page_bytes = int(meta["rows_per_page"]), int(meta["page_bytes"])
    if rows_per_page * row_bytes > page_bytes:
        raise ValueError("PLE rows do not fit in their pages")
    expected = -(-rows // rows_per_page) * page_bytes
    if data.stat().st_size != expected:
        raise ValueError(f"{data} has {data.stat().st_size} bytes, expected {expected}")
    meta["data_path"] = str(data)
    return meta


class Qwen4ExpPLESSDEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """PLE table read on demand from a row file instead of pinned memory.

    Reuses the pinned backend's side-stream prefetch and finalization; only the
    row lookup differs. The checkpoint rows are never materialized: the weight
    parameter lives on the meta device, so any accidental direct use fails.
    """

    supports_prefetch = True
    skips_checkpoint_rows = True

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        padding_size: int,
        prefix: str,
        embedding_method: Qwen4ExpPLEEmbeddingMethod,
        num_ngram_heads: int = 1,
        max_total_tokens: int = 0,
        data_parallel_rank: int = 0,
        ssd_rows_path: str,
        reader_threads: int = 16,
    ) -> None:
        if not isinstance(embedding_method, Qwen4ExpPLEFp8EmbeddingMethod):
            raise NotImplementedError("SSD PLE rows require an FP8 PLE checkpoint")
        # Skip the pinned backend's constructor: it would pin the full table.
        Qwen4ExpPLEEmbedding.__init__(
            self,
            num_embeddings,
            embedding_dim,
            params_dtype=params_dtype,
            padding_size=padding_size,
            prefix=prefix,
            embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads,
            max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank,
        )
        if self.tp_size != 1:
            raise NotImplementedError("SSD PLE rows currently require ETP size 1")
        meta = load_row_file_metadata(ssd_rows_path)
        if int(meta["row_bytes"]) != self.embedding_dim:
            raise ValueError(
                f"PLE row file has {meta['row_bytes']}-byte rows, model expects "
                f"{self.embedding_dim}")
        vocab_end = self.shard_indices.org_vocab_end_index
        if int(meta["rows"]) < vocab_end:
            raise ValueError(
                f"PLE row file has {meta['rows']} rows, model addresses {vocab_end}")
        self._meta = meta
        self._num_rows = int(meta["rows"])
        self._row_bytes = int(meta["row_bytes"])
        self._rows_per_page = int(meta["rows_per_page"])
        self._page_bytes = int(meta["page_bytes"])
        self._reader_threads = int(reader_threads)
        self._fd = os.open(meta["data_path"], os.O_RDONLY)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(self._fd, 0, 0, os.POSIX_FADV_RANDOM)
        # The stream-ordered path needs the extension; there is no slow fallback.
        self._ext = load_reader_module()
        self._params: dict[int, int] = {}
        self._scale_checked = False

        device = torch.device("cuda", torch.cuda.current_device())
        capacity = max(1, max_total_tokens * self.etp_data_parallel_size)
        self._prefetch_stream = torch.cuda.Stream(device=device)
        self._prefetch_buffer = torch.empty(
            capacity,
            num_ngram_heads,
            self.embedding_dim,
            dtype=torch.float8_e4m3fn,
            device=device,
        )
        self._output_dim = num_ngram_heads * self.embedding_dim
        self._alloc_host_buffers(capacity * num_ngram_heads)
        logger.info(
            "Serving PLE rows from %s: %d rows, %.1f GiB on disk, read through the "
            "page cache; the pinned host table is not allocated.",
            meta["data_path"], self._num_rows, meta["bytes"] / 2**30)

    def _alloc_host_buffers(self, rows: int) -> None:
        self._params = {}  # parameter blocks point into the old buffers
        # Explicit device: model construction runs under a CUDA default device.
        self._host_ids = torch.empty(
            rows, dtype=torch.int64, device="cpu", pin_memory=True)
        self._host_rows = torch.empty(
            rows, self._row_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)

    def allocate_embedding_weight(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Describe the table shape without allocating it."""
        return torch.empty(num_embeddings, embedding_dim, dtype=dtype, device="meta")

    def _check_scale(self) -> None:
        scale = float(self.weight_scale.reshape(-1)[0].item())
        if abs(scale - float(self._meta["scale"])) > 1e-12 * max(1.0, abs(scale)):
            raise ValueError(
                f"PLE row file scale {self._meta['scale']} does not match the "
                f"checkpoint scale {scale}; re-export the row file")
        self._scale_checked = True

    def _lookup(
        self,
        input_ids: torch.Tensor,
        output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Read the requested FP8 rows from disk into ``output``, in stream order.

        Device-to-host copy of the IDs, a CUDA host function that reads the rows,
        and a host-to-device copy of the rows: all three are ordinary stream
        operations, so the lookup is also valid inside a captured CUDA graph.
        """
        expected_shape = (*input_ids.shape, self.embedding_dim)
        if output is None:
            output = torch.empty(
                expected_shape, dtype=torch.float8_e4m3fn, device=input_ids.device)
        elif tuple(output.shape) != expected_shape or output.dtype != torch.float8_e4m3fn:
            raise ValueError("PLE prefetch output must match the input shape and FP8 dtype")
        flat_ids = input_ids.reshape(-1)
        count = flat_ids.numel()
        if count == 0:
            return output
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            if not self._scale_checked:
                self._check_scale()
            errors = self._ext.take_errors()
            if errors:
                raise RuntimeError(f"Failed to read {errors} PLE rows from disk")
            if count > self._host_ids.numel():
                torch.cuda.current_stream().synchronize()
                self._alloc_host_buffers(count)
        elif count > self._host_ids.numel():
            raise RuntimeError("PLE SSD staging is too small for a captured batch")
        params = self._params.get(count)
        if params is None:
            params = self._ext.make_params(
                self._fd, self._host_ids, self._host_rows, count, self._row_bytes,
                self._rows_per_page, self._page_bytes, self._num_rows,
                self._reader_threads)
            self._params[count] = params
        stream = torch.cuda.current_stream()
        self._host_ids[:count].copy_(flat_ids, non_blocking=True)
        self._ext.enqueue_gather(stream.cuda_stream, params)
        output.view(count, self.embedding_dim).view(torch.uint8).copy_(
            self._host_rows[:count], non_blocking=True)
        return output

    def __del__(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
