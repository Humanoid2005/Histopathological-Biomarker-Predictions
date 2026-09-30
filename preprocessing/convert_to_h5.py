"""
PNG → HDF5 packer, parallel-reader version.

Why this is faster than v1 on a spinning HDD:
  - N concurrent file reads keep the disk's internal queue full, which lets
    the drive firmware reorder seeks. A single-threaded reader can't do that.
  - HDF5 write path is unchanged (batched 64 MB contiguous writes, one thread).
"""
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import h5py
import numpy as np
from tqdm import tqdm

from config import PATCHES_PATH

WRITE_CHUNK   = 64 * 1024 * 1024
EXTS          = (".png", ".jpg", ".jpeg")
READ_WORKERS  = 32      # try 16 / 32 / 64 — 32 is a good default for one HDD
IN_FLIGHT     = 4 * READ_WORKERS   # bounded window → caps RAM


def _read(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def slide_to_hdf5(slide_dir: str, output_file: str) -> bool:
    # ---- Pass 1: scandir + stat ------------------------------------------- #
    entries = []
    with os.scandir(slide_dir) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(EXTS):
                try:
                    st = e.stat()
                except OSError:
                    continue
                entries.append((e.path, e.name, st.st_size))

    if not entries:
        return False

    entries.sort(key=lambda x: x[1])
    n = len(entries)

    sizes = np.fromiter((s for _, _, s in entries), dtype=np.int64, count=n)
    offsets = np.empty(n, dtype=np.int64)
    offsets[0] = 0
    if n > 1:
        np.cumsum(sizes[:-1], out=offsets[1:])
    total = int(offsets[-1]) + int(sizes[-1])

    lengths = sizes.astype(np.uint32) if sizes.max() < 2**32 else sizes.astype(np.int64)
    names   = np.array([nm for _, nm, _ in entries], dtype=h5py.string_dtype())

    # ---- Pass 2: parallel reads, single-threaded HDF5 writes -------------- #
    with h5py.File(output_file, "w", libver="latest") as f:
        buf = f.create_dataset("image_bytes", shape=(total,), dtype=np.uint8)

        acc = bytearray()
        acc_start = 0

        with ThreadPoolExecutor(max_workers=READ_WORKERS) as ex:
            pending = deque()
            head = 0
            # Prime the pipeline
            while head < n and len(pending) < IN_FLIGHT:
                pending.append(ex.submit(_read, entries[head][0]))
                head += 1

            for _ in tqdm(range(n), desc=os.path.basename(slide_dir),
                          total=n, leave=True, dynamic_ncols=True,
                          mininterval=1.0):
                data = pending.popleft().result()

                # Refill the window
                if head < n:
                    pending.append(ex.submit(_read, entries[head][0]))
                    head += 1

                acc.extend(data)
                if len(acc) >= WRITE_CHUNK:
                    end = acc_start + len(acc)
                    buf[acc_start:end] = np.frombuffer(acc, dtype=np.uint8)
                    acc_start = end
                    acc.clear()

            if acc:
                end = acc_start + len(acc)
                buf[acc_start:end] = np.frombuffer(acc, dtype=np.uint8)

        f.create_dataset("offsets",   data=offsets)
        f.create_dataset("lengths",   data=lengths)
        f.create_dataset("filenames", data=names)

    return True


def main():
    base_dir = PATCHES_PATH
    out_dir  = os.path.join(os.path.dirname(base_dir), "h5_patches")
    os.makedirs(out_dir, exist_ok=True)

    slides = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    )
    print(f"Found {len(slides)} slides in {base_dir}")

    for name in tqdm(slides, desc="Slides", dynamic_ncols=True, mininterval=1.0):
        out_file = os.path.join(out_dir, f"{name}.h5")
        if os.path.exists(out_file):
            tqdm.write(f"  Skipping {name} (already done)")
            continue
        tqdm.write(f"  Processing {name} ...")
        if not slide_to_hdf5(os.path.join(base_dir, name), out_file):
            tqdm.write(f"  No patches found in {name}, skipped.")

    print(f"\nDone. Files in: {out_dir}")


if __name__ == "__main__":
    main()