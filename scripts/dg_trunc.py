#!/usr/bin/env python3
"""Layer-truncated copy of a remote DiffusionGemma GGUF, fetched with HTTP range requests.

Keeps the first --layers blocks plus every non-block tensor, rewrites block_count and the per-layer
arrays, and never downloads the dropped layers. Usage:
  PYTHONPATH=<llama.cpp>/gguf-py python dg_trunc.py URL OUT.gguf --layers 6 [--work DIR]
"""
import argparse
import os
import re
import sys
import urllib.request

import numpy as np
import gguf


def fetch(url, start, end, f, chunk=64 << 20):
    pos = start
    while pos < end:
        stop = min(end, pos + chunk)
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={pos}-{stop - 1}"})
                with urllib.request.urlopen(req, timeout=300) as r:
                    data = r.read()
                break
            except Exception as e:  # noqa: BLE001
                print(f"retry {attempt} {pos}: {e}", file=sys.stderr)
        else:
            raise SystemExit(f"range {pos}-{stop} failed")
        assert len(data) == stop - pos, (len(data), stop - pos)
        f.seek(pos)
        f.write(data)
        pos = stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--keep", default=None, help="comma list of source layer ids to keep (renumbered 0..); overrides --layers")
    ap.add_argument("--work", default=None)
    ap.add_argument("--header-mb", type=int, default=48)
    a = ap.parse_args()

    req = urllib.request.Request(a.url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        size = int(r.headers["Content-Length"])
    work = a.work or os.path.dirname(os.path.abspath(a.out))
    sparse = os.path.join(work, "src_sparse.gguf")
    with open(sparse, "wb") as f:
        f.truncate(size)
        fetch(a.url, 0, min(size, a.header_mb << 20), f)

    reader = gguf.GGUFReader(sparse, "r")
    arch = reader.fields["general.architecture"].contents()
    nl_key = f"{arch}.block_count"
    n_layer = reader.fields[nl_key].contents()
    ids = [int(x) for x in a.keep.split(",")] if a.keep else list(range(a.layers))
    keep = len(ids)
    remap = {src: dst for dst, src in enumerate(ids)}
    print(f"arch={arch} layers {n_layer} -> {ids}")

    blk = re.compile(r"^blk\.(\d+)\.")

    def wanted(t):
        m = blk.match(t.name)
        return not m or int(m.group(1)) in remap

    def new_name(name):
        m = blk.match(name)
        return name if not m else f"blk.{remap[int(m.group(1))]}." + name[m.end():]

    kept = [t for t in reader.tensors if wanted(t)]
    with open(sparse, "r+b") as f:
        for t in kept:
            if t.data_offset + t.n_bytes > (a.header_mb << 20):
                fetch(a.url, t.data_offset, t.data_offset + t.n_bytes, f)
    reader = gguf.GGUFReader(sparse, "r")
    kept = [t for t in reader.tensors if wanted(t)]

    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = field.contents()
        if field.name == nl_key:
            val = keep
        elif vt == gguf.GGUFValueType.ARRAY and field.name.startswith(arch + ".") and isinstance(val, list) and len(val) == n_layer:
            val = [val[i] for i in ids]
            print(f"truncate per-layer array {field.name} -> {val}")
        writer.add_key_value(field.name, val, vt, sub_type=st)
    for t in kept:
        writer.add_tensor_info(new_name(t.name), t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    del reader
    os.remove(sparse)
    print(f"wrote {a.out} {os.path.getsize(a.out) / 1e9:.2f} GB, {len(kept)} tensors")


if __name__ == "__main__":
    main()
