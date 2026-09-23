#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export the Qwen3.8-Flash-Next PLE n-gram table as a plain, page-aligned row file for SSD lookup.

Layout (little-endian, no header):
  global row r  ->  byte offset (r // ROWS_PER_PAGE) * PAGE_BYTES + (r % ROWS_PER_PAGE) * ROW_BYTES
  ROW_BYTES = 160 (FP8 E4M3, one 160-dim head vector), ROWS_PER_PAGE = 25, PAGE_BYTES = 4096.
  Each page holds 25 rows followed by 96 zero bytes, so a row never crosses a 4 KiB boundary and
  one O_DIRECT 4 KiB read returns it. Dequantize: value = fp8(row) * scale (scale in the sidecar).

Global row order matches vLLM's loader (ngram_embedding.py, load_weights): checkpoint shard i fills
rows [i * shard_size, i * shard_size + rows_i) with shard_size = ceil(total_rows / split_ngram_parts).

Standard library only (numpy used when available for speed). Streams; memory stays below ~100 MiB.
"""
import argparse, hashlib, json, os, random, struct, sys, time
from pathlib import Path

ROW_BYTES, ROWS_PER_PAGE, PAGE_BYTES = 160, 25, 4096
PREFIX = 'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.'

try:
    import numpy as np
except ImportError:  # pragma: no cover - pure-Python fallback
    np = None


def is_prime_64(value):
    """Deterministic Miller-Rabin, identical to vLLM's _is_prime_64."""
    if value < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % p == 0:
            return value == p
    d, s = value - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, value)
            if x == value - 1:
                break
        else:
            return False
    return True


def nth_prime_after(start, count):
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


def vocab_layout(base, heads, dense_layer_id):
    sizes, offsets, offset = [], [], 0
    for local in range(heads):
        size = nth_prime_after(base - 1, dense_layer_id * heads + local + 1)
        sizes.append(size)
        offsets.append(offset)
        offset += size
    return sizes, offsets, offset


def read_header(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def bf16_to_float(raw):
    return struct.unpack('<f', b'\x00\x00' + raw)[0]


def pack_pages(rows_bytes):
    """rows_bytes holds a multiple of 25 rows; return the page-aligned bytes."""
    pages = len(rows_bytes) // (ROW_BYTES * ROWS_PER_PAGE)
    if np is not None:
        src = np.frombuffer(rows_bytes, dtype=np.uint8).reshape(pages, ROW_BYTES * ROWS_PER_PAGE)
        out = np.zeros((pages, PAGE_BYTES), dtype=np.uint8)
        out[:, :ROW_BYTES * ROWS_PER_PAGE] = src
        return out.tobytes()
    pad = bytes(PAGE_BYTES - ROW_BYTES * ROWS_PER_PAGE)
    step = ROW_BYTES * ROWS_PER_PAGE
    return b''.join(rows_bytes[i:i + step] + pad for i in range(0, len(rows_bytes), step))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='checkpoint directory')
    ap.add_argument('--output', required=True, help='output directory')
    ap.add_argument('--verify-samples', type=int, default=200_000)
    args = ap.parse_args()
    model, out_dir = Path(args.model), Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path, meta_path = out_dir / 'ple_rows.fp8.bin', out_dir / 'ple_rows.json'
    if data_path.exists():
        sys.exit(f'{data_path} exists; remove it first')

    config = json.loads((model / 'config.json').read_text())
    text = config.get('text_config', config)
    heads = (int(text['ngram_size']) - 1) * int(text['heads_per_ngram'])
    parts = int(text['split_ngram_parts'])
    row_dim = int(text['ple_embed_dim']) // heads
    if row_dim != ROW_BYTES:
        sys.exit(f'unexpected row width {row_dim}')

    index = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
    shard_names = sorted((k for k in index if k.startswith(PREFIX + 'shard_') and k.endswith('.weight')),
                         key=lambda k: int(k[len(PREFIX + 'shard_'):-len('.weight')]))
    headers = {}
    shards = []
    for name in shard_names:
        file = model / index[name]
        if file not in headers:
            headers[file] = read_header(file)
        header, data_start = headers[file]
        t = header[name]
        if t['dtype'] != 'F8_E4M3' or t['shape'][1] != ROW_BYTES:
            sys.exit(f'unexpected tensor {name}: {t}')
        begin, end = t['data_offsets']
        shards.append(dict(name=name, file=file, offset=data_start + begin, rows=t['shape'][0], nbytes=end - begin))
    total_rows = sum(s['rows'] for s in shards)
    shard_size = -(-total_rows // parts)

    # Cross-check against vLLM's prime layout for the dense PLE layer id that reproduces the checkpoint.
    # The checkpoint stores the table padded to make_ngram_vocab_size_divisible_by; rows past the
    # real total are padding and are never addressed.
    divisor = int(text['make_ngram_vocab_size_divisible_by'])
    layout = None
    for dense_id in range(4):
        sizes, offsets, total = vocab_layout(int(text['ngram_vocab_size_base']), heads, dense_id)
        if -(-total // divisor) * divisor == total_rows:
            layout = dict(ple_dense_layer_id=dense_id, head_vocab_sizes=sizes, head_offsets=offsets,
                          addressable_rows=total, padding_rows=total_rows - total)
            break
    if layout is None:
        sys.exit(f'no prime layout reproduces {total_rows} rows')
    if len(shards) != parts or any(s['rows'] != shard_size for s in shards[:-1]):
        sys.exit('shard sizes do not match split_ngram_parts')

    scale_name = PREFIX + 'weight_scale'
    header, data_start = headers.setdefault(model / index[scale_name], read_header(model / index[scale_name]))
    st = header[scale_name]
    with open(model / index[scale_name], 'rb') as f:
        f.seek(data_start + st['data_offsets'][0])
        raw = f.read(st['data_offsets'][1] - st['data_offsets'][0])
    scale = bf16_to_float(raw) if st['dtype'] == 'BF16' else struct.unpack('<f', raw)[0]

    print(f'{total_rows:,} rows in {len(shards)} shards, shard_size {shard_size:,}, scale {scale!r}, '
          f'dense layer id {layout["ple_dense_layer_id"]}', flush=True)

    batch_rows = ROWS_PER_PAGE * 4096           # 102,400 rows = 16 MB of input per write
    digest, pending, written_rows = hashlib.sha256(), bytearray(), 0
    shard_sha = []
    t0 = time.time()
    with open(data_path, 'wb', buffering=0) as out:
        for i, s in enumerate(shards):
            h = hashlib.sha256()
            with open(s['file'], 'rb', buffering=0) as src:
                src.seek(s['offset'])
                remaining = s['nbytes']
                while remaining:
                    chunk = src.read(min(remaining, batch_rows * ROW_BYTES))
                    if not chunk:
                        sys.exit(f'short read in {s["name"]}')
                    remaining -= len(chunk)
                    h.update(chunk)
                    pending += chunk
                    full = len(pending) // (ROW_BYTES * ROWS_PER_PAGE) * (ROW_BYTES * ROWS_PER_PAGE)
                    if full:
                        page_bytes = pack_pages(bytes(pending[:full]))
                        out.write(page_bytes)
                        digest.update(page_bytes)
                        written_rows += full // ROW_BYTES
                        del pending[:full]
                try:
                    os.posix_fadvise(src.fileno(), s['offset'], s['nbytes'], os.POSIX_FADV_DONTNEED)
                except (AttributeError, OSError):
                    pass
            shard_sha.append(h.hexdigest())
            if i % 8 == 7 or i == len(shards) - 1:
                done = (i + 1) / len(shards)
                rate = written_rows * ROW_BYTES / max(1e-9, time.time() - t0) / 1e6
                print(f'  shard {i + 1}/{len(shards)}  {done:5.1%}  {rate:,.0f} MB/s', flush=True)
        if pending:  # last partial page: remaining rows, zero padding
            tail = bytes(pending) + bytes(PAGE_BYTES - len(pending))
            out.write(tail)
            digest.update(tail)
            written_rows += len(pending) // ROW_BYTES
        out.flush()
        os.fsync(out.fileno())
    pages = -(-total_rows // ROWS_PER_PAGE)
    size = data_path.stat().st_size
    if written_rows != total_rows or size != pages * PAGE_BYTES:
        sys.exit(f'size check failed: rows {written_rows} / {total_rows}, bytes {size} / {pages * PAGE_BYTES}')

    # Independent verification: random rows plus every shard boundary, read back as whole 4 KiB pages.
    rng = random.Random(20260923)
    rows = {0, total_rows - 1}
    for i in range(len(shards)):
        rows.update({i * shard_size, min(total_rows, (i + 1) * shard_size) - 1})
    while len(rows) < args.verify_samples:
        rows.add(rng.randrange(total_rows))
    mismatches = 0
    with open(data_path, 'rb') as dst:
        handles = {}
        for r in sorted(rows):
            s = shards[r // shard_size]
            src = handles.setdefault(s['file'], open(s['file'], 'rb'))
            expected = os.pread(src.fileno(), ROW_BYTES, s['offset'] + (r % shard_size) * ROW_BYTES)
            page = os.pread(dst.fileno(), PAGE_BYTES, (r // ROWS_PER_PAGE) * PAGE_BYTES)
            got = page[(r % ROWS_PER_PAGE) * ROW_BYTES:(r % ROWS_PER_PAGE + 1) * ROW_BYTES]
            mismatches += expected != got
        for h in handles.values():
            h.close()

    meta = dict(
        format='qwen4-exp-ple-rows', version=1,
        file=data_path.name, bytes=size, sha256=digest.hexdigest(),
        rows=total_rows, row_bytes=ROW_BYTES, rows_per_page=ROWS_PER_PAGE, page_bytes=PAGE_BYTES,
        pages=pages, dtype='float8_e4m3fn', scale=scale, scale_source_dtype=st['dtype'],
        row_offset='(row // rows_per_page) * page_bytes + (row % rows_per_page) * row_bytes',
        dequantize='float(fp8) * scale',
        vllm_row_order='shard i -> rows [i*shard_size, i*shard_size + rows_i)',
        shard_size=shard_size, split_ngram_parts=parts, ngram_heads=heads,
        ngram_size=int(text['ngram_size']), heads_per_ngram=int(text['heads_per_ngram']),
        ngram_vocab_size_base=int(text['ngram_vocab_size_base']), **layout,
        source=dict(model_dir=str(model), files=sorted({s['file'].name for s in shards}),
                    tensor_prefix=PREFIX, shard_sha256=shard_sha),
        verification=dict(sampled_rows=len(rows), mismatches=mismatches,
                          includes='every shard first/last row plus random rows'),
        created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        elapsed_s=round(time.time() - t0, 1),
    )
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ('bytes', 'rows', 'pages', 'scale', 'sha256', 'verification', 'elapsed_s')}))
    if mismatches:
        sys.exit(f'{mismatches} sampled rows differ')


if __name__ == '__main__':
    main()
