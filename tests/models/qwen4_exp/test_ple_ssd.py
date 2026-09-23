# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the SSD-backed Qwen4Exp PLE row file and reader."""

import json
import os

import pytest
import torch

from vllm.config.engram import EngramConfig
from vllm.models.qwen4_exp.nvidia.ple_ssd import (
    _python_gather_rows,
    load_reader_module,
    load_row_file_metadata,
    load_row_reader,
)

ROW_BYTES, ROWS_PER_PAGE, PAGE_BYTES = 160, 25, 4096


def _write_row_file(tmp_path, rows: torch.Tensor) -> str:
    """Write rows with the exporter's page-aligned layout and return the metadata path."""
    num_rows = rows.shape[0]
    pages = -(-num_rows // ROWS_PER_PAGE)
    data = torch.zeros(pages, PAGE_BYTES, dtype=torch.uint8)
    for r in range(num_rows):
        start = (r % ROWS_PER_PAGE) * ROW_BYTES
        data[r // ROWS_PER_PAGE, start:start + ROW_BYTES] = rows[r]
    (tmp_path / "ple_rows.fp8.bin").write_bytes(data.numpy().tobytes())
    meta = dict(format="qwen4-exp-ple-rows", version=1, file="ple_rows.fp8.bin",
                bytes=pages * PAGE_BYTES, rows=num_rows, row_bytes=ROW_BYTES,
                rows_per_page=ROWS_PER_PAGE, page_bytes=PAGE_BYTES, pages=pages,
                dtype="float8_e4m3fn", scale=0.5)
    path = tmp_path / "ple_rows.json"
    path.write_text(json.dumps(meta))
    return str(path)


@pytest.fixture
def row_file(tmp_path):
    gen = torch.Generator().manual_seed(0)
    rows = torch.randint(0, 256, (1003, ROW_BYTES), dtype=torch.uint8, generator=gen)
    return rows, load_row_file_metadata(_write_row_file(tmp_path, rows))


@pytest.mark.parametrize("use_extension", [True, False])
@pytest.mark.parametrize("count", [7, 5000])
def test_gather_rows_matches_layout(row_file, use_extension, count):
    rows, meta = row_file
    if use_extension:
        try:
            reader = load_row_reader(allow_fallback=False)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"C++ row reader cannot be built here: {exc}")
    else:
        reader = _python_gather_rows
    gen = torch.Generator().manual_seed(count)
    ids = torch.randint(0, rows.shape[0], (count,), dtype=torch.int64, generator=gen)
    ids[0], ids[-1] = rows.shape[0] - 1, 0          # last and first row
    ids[count // 2] = -1                             # masked: other ETP rank
    ids[count // 3] = rows.shape[0]                  # masked: out of range
    out = torch.full((count, ROW_BYTES), 7, dtype=torch.uint8)
    fd = os.open(meta["data_path"], os.O_RDONLY)
    try:
        errors = reader(fd, ids, out, ROW_BYTES, ROWS_PER_PAGE, PAGE_BYTES,
                        rows.shape[0], 8)
    finally:
        os.close(fd)
    assert errors == 0
    valid = (ids >= 0) & (ids < rows.shape[0])
    torch.testing.assert_close(out[valid], rows[ids[valid]])
    assert torch.all(out[~valid] == 0)


def test_metadata_rejects_truncated_file(tmp_path):
    rows = torch.zeros(60, ROW_BYTES, dtype=torch.uint8)
    path = _write_row_file(tmp_path, rows)
    data = tmp_path / "ple_rows.fp8.bin"
    data.write_bytes(data.read_bytes()[:-1])
    with pytest.raises(ValueError, match="expected"):
        load_row_file_metadata(path)


def test_metadata_rejects_other_formats(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text(json.dumps({"format": "something-else", "version": 1}))
    with pytest.raises(ValueError, match="not a version-1"):
        load_row_file_metadata(str(path))


def test_engram_config_requires_cpu_offload():
    with pytest.raises(ValueError, match="ssd_rows_path requires cpu_offload"):
        EngramConfig(cpu_offload=False, ssd_rows_path="/tmp/ple_rows.json")
    assert EngramConfig(cpu_offload=True, ssd_rows_path="/tmp/x.json").ssd_reader_threads == 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.parametrize("captured", [False, True])
def test_stream_ordered_gather_in_cuda_graph(row_file, captured):
    """D2H IDs -> host-function pread -> H2D rows, eagerly and replayed from a graph."""
    rows, meta = row_file
    ext = load_reader_module()
    ext.take_errors()
    n = 64
    host_ids = torch.empty(n, dtype=torch.int64, pin_memory=True)
    host_rows = torch.empty(n, ROW_BYTES, dtype=torch.uint8, pin_memory=True)
    fd = os.open(meta["data_path"], os.O_RDONLY)
    params = ext.make_params(fd, host_ids, host_rows, n, ROW_BYTES, ROWS_PER_PAGE,
                             PAGE_BYTES, rows.shape[0], 4)
    dev_ids = torch.zeros(n, dtype=torch.int64, device="cuda")
    out = torch.empty(n, ROW_BYTES, dtype=torch.uint8, device="cuda")
    stream = torch.cuda.Stream()

    def step():
        host_ids.copy_(dev_ids, non_blocking=True)
        ext.enqueue_gather(torch.cuda.current_stream().cuda_stream, params)
        out.copy_(host_rows, non_blocking=True)

    graph = None
    if captured:
        with torch.cuda.stream(stream):
            step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            step()
    try:
        for seed in (1, 2, 3):
            gen = torch.Generator().manual_seed(seed)
            ids = torch.randint(0, rows.shape[0], (n,), dtype=torch.int64, generator=gen)
            dev_ids.copy_(ids.cuda())
            if graph is not None:
                graph.replay()
            else:
                with torch.cuda.stream(stream):
                    step()
            torch.cuda.synchronize()
            torch.testing.assert_close(out.cpu(), rows[ids])
    finally:
        os.close(fd)
    assert ext.take_errors() == 0
