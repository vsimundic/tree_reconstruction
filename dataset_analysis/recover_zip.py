#!/usr/bin/env python3
"""Recover files from a truncated/index-less streaming zip.

Archive_Filtered.zip has no central directory (last ~1GB is zero-padded) and
uses data descriptors, so normal unzip fails. This walks local file headers
sequentially, decompressing each deflate stream until it ends, then locating
the data descriptor to find the next entry.
"""
import os, sys, struct, zlib, argparse

SRC_DEFAULT = "/workspace/data/Archive_Filtered.zip"
DATA_END = 33901940736  # start of trailing zero region

def read_local_header(f, pos):
    f.seek(pos)
    sig = f.read(4)
    if sig != b'PK\x03\x04':
        return None
    ver, flags, method, mt, md, crc, csize, usize, nlen, elen = struct.unpack("<HHHHHIIIHH", f.read(26))
    name = f.read(nlen).decode("utf-8", "replace")
    f.read(elen)  # skip extra
    data_start = pos + 30 + nlen + elen
    return dict(flags=flags, method=method, crc=crc, csize=csize, usize=usize,
                name=name, data_start=data_start)

def extract(src, out, limit, dry):
    f = open(src, "rb")
    pos = 0
    n = 0
    while pos < DATA_END:
        h = read_local_header(f, pos)
        if h is None:
            print(f"[stop] no local header at {pos}")
            break
        name = h["name"]
        is_dir = name.endswith("/")
        has_dd = bool(h["flags"] & 0x08)
        dest = os.path.join(out, name)
        if is_dir:
            if not dry: os.makedirs(dest, exist_ok=True)
            pos = h["data_start"]
            n += 1
            continue

        f.seek(h["data_start"])
        if h["method"] == 0 and not has_dd:
            csize = h["csize"]; usize = csize
            if not dry:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                w = open(dest, "wb")
            remaining = csize
            crc = 0
            while remaining:
                b = f.read(min(1 << 20, remaining))
                if not b: break
                remaining -= len(b); crc = zlib.crc32(b, crc)
                if not dry: w.write(b)
            if not dry: w.close()
            comp_consumed = csize
        elif h["method"] == 8:
            d = zlib.decompressobj(-15)
            if not dry:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                w = open(dest, "wb")
            usize = 0; crc = 0; fed = 0
            while not d.eof:
                b = f.read(1 << 20)
                if not b:
                    print(f"[warn] EOF mid-stream in {name}"); break
                fed += len(b)
                out_chunk = d.decompress(b)
                usize += len(out_chunk); crc = zlib.crc32(out_chunk, crc)
                if not dry: w.write(out_chunk)
            if not dry: w.close()
            comp_consumed = fed - len(d.unused_data)
        else:
            print(f"[stop] unsupported method {h['method']} in {name}")
            break

        # locate next entry
        after = h["data_start"] + comp_consumed
        if has_dd:
            f.seek(after)
            dd = f.read(24)
            off = 4 if dd[:4] == b'PK\x07\x08' else 0
            # disambiguate 4- vs 8-byte sizes using our known usize
            u32 = struct.unpack("<I", dd[off+8:off+12])[0]
            if u32 == (usize & 0xffffffff):
                dd_len = off + 12
            else:
                dd_len = off + 20  # zip64
            after += dd_len
        expected = h["crc"] if not has_dd else None
        ok = "" if (expected is None or expected == crc) else " CRC-MISMATCH"
        print(f"[ok] {name}  usize={usize}{ok}")
        pos = after
        n += 1
        if limit and n >= limit:
            print(f"[limit] stopped after {n} entries at pos {pos}")
            break
    print(f"[done] processed {n} entries, final pos={pos}, data_end={DATA_END}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default="/workspace/data/Archive_Filtered_recovered")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    extract(a.src, a.out, a.limit, a.dry)
