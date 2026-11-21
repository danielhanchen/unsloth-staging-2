#!/usr/bin/env python3
"""Copy a GGUF adding ``general.architecture`` (string) when it is missing
tensor data copied byte for byte.

  python gguf_set_arch.py SRC DST ARCH

unsloth/Qwen-Image-2.1-GGUF ships no general.architecture key (its Turbo sibling says "qwen_image"), and
the GGUF loader node pack refuses an image GGUF without one unless its own key detector knows the layout (it has no Qwen-Image
entry). Tensor infos keep their offsets (relative to the data section), so only the header and its padding change.
"""
import shutil
import struct
import sys

SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def main():
    src, dst, arch = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src, "rb") as f:
        head = f.read(64 << 20)
    o = 0

    def rd(fmt):
        nonlocal o
        v = struct.unpack_from("<" + fmt, head, o)
        o += struct.calcsize("<" + fmt)
        return v[0]

    def skip_val(t):
        nonlocal o
        if t == 8:
            n = rd("Q")
            o += n
        elif t == 9:
            at = rd("I")
            n = rd("Q")
            for _ in range(n):
                skip_val(at)
        else:
            o += SZ[t]

    assert head[:4] == b"GGUF"
    o = 4
    ver = rd("I")
    nt = rd("Q")
    nkv = rd("Q")
    kv_start = o
    align = 32
    keys = []
    for _ in range(nkv):
        n = rd("Q")
        k = head[o:o + n].decode()
        o += n
        t = rd("I")
        if k == "general.alignment":
            align = struct.unpack_from("<I", head, o)[0]
        skip_val(t)
        keys.append(k)
    kv_end = o
    for _ in range(nt):
        n = rd("Q")
        o += n
        nd = rd("I")
        o += 8 * nd + 4 + 8
    info_end = o
    data_start = (info_end + align - 1) // align * align
    if "general.architecture" in keys:
        print("ARCH_PRESENT", src)
        shutil.copyfile(src, dst)
        return 0
    kb = b"general.architecture"
    vb = arch.encode()
    new_kv = struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb
    hdr = head[:4] + struct.pack("<IQQ", ver, nt, nkv + 1) + new_kv + head[kv_start:kv_end] + head[kv_end:info_end]
    pad = (-len(hdr)) % align
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        fo.write(hdr + b"\0" * pad)
        fi.seek(data_start)
        shutil.copyfileobj(fi, fo, 64 << 20)
    print("ARCH_SET", dst, arch, "kv", nkv, "->", nkv + 1, "tensors", nt, "data", data_start, "->", len(hdr) + pad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
