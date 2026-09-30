"""
PNG → HDF5 packer using direct SMB streaming.
- Reads patches directly from Input NAS via smbclient.
- Builds the HDF5 file locally (fast, avoids network HDF5 corruption).
- Uploads the final .h5 file directly to Output NAS via smbclient.
"""
import os
import tempfile
import shutil
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

import h5py
import numpy as np
from tqdm import tqdm

# --- Configuration ---
import glob
uid = os.getuid()

def get_gvfs_path(server_ip, share_name, subpath):
    try:
        server_dir = glob.glob(f"/run/user/{uid}/gvfs/smb-share:server={server_ip}*")[0]
        if os.path.exists(os.path.join(server_dir, share_name)):
            return os.path.join(server_dir, share_name, subpath)
        else:
            return os.path.join(server_dir, subpath)
    except IndexError:
        return f"/run/user/{uid}/gvfs/MISSING_{server_ip}"

BASE_IN_SMB = get_gvfs_path("172.16.201.2", "prof-sushree", "sriram-srikanth/patches")
BASE_OUT_SMB = get_gvfs_path("172.16.202.70", "home", "sriram-srikanth/patches")

WRITE_CHUNK = 64 * 1024 * 1024
EXTS = (".png", ".jpg", ".jpeg")
READ_WORKERS = 8
IN_FLIGHT = 16

def _read_smb(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

def safe_scandir_smb(slide_dir):
    entries = []
    try:
        for name in os.listdir(slide_dir):
            path = os.path.join(slide_dir, name)
            if os.path.isfile(path) and name.lower().endswith(EXTS):
                entries.append((path, name))
    except Exception as e:
        tqdm.write(f"  [!] scandir failed for {slide_dir}: {e}")
        return []
    return entries

def process_slide(slide_dir: str, local_h5_path: str) -> bool:
    entries = safe_scandir_smb(slide_dir)
    if not entries:
        return False

    entries.sort(key=lambda x: x[1])
    n = len(entries)

    with h5py.File(local_h5_path, "w", libver="latest") as f:
        buf = f.create_dataset("image_bytes", shape=(0,), maxshape=(None,), 
                               dtype=np.uint8, chunks=(4 * 1024 * 1024,))

        offsets = []
        lengths = []
        names = []

        acc = bytearray()
        current_offset = 0

        ex = ThreadPoolExecutor(max_workers=READ_WORKERS)
        try:
            pending = deque()
            head = 0
            while head < n and len(pending) < IN_FLIGHT:
                path, name = entries[head]
                pending.append((name, ex.submit(_read_smb, path)))
                head += 1

            for _ in tqdm(range(n), desc=os.path.basename(slide_dir),
                          total=n, leave=False, position=1, dynamic_ncols=True,
                          mininterval=1.0):
                name, fut = pending.popleft()

                if head < n:
                    path, next_name = entries[head]
                    pending.append((next_name, ex.submit(_read_smb, path)))
                    head += 1

                try:
                    data = fut.result(timeout=15.0)
                except FuturesTimeoutError:
                    tqdm.write(f"  [!] Timeout reading {name}, skipping file.")
                    continue
                except Exception as e:
                    tqdm.write(f"  [!] Failed reading {name} ({e}), skipping file.")
                    continue

                length = len(data)
                if length == 0:
                    continue

                offsets.append(current_offset)
                lengths.append(length)
                names.append(name)

                acc.extend(data)
                current_offset += length

                if len(acc) >= WRITE_CHUNK:
                    old_len = buf.shape[0]
                    new_len = old_len + len(acc)
                    buf.resize((new_len,))
                    buf[old_len:new_len] = np.frombuffer(acc, dtype=np.uint8)
                    acc.clear()

            if acc:
                old_len = buf.shape[0]
                new_len = old_len + len(acc)
                buf.resize((new_len,))
                buf[old_len:new_len] = np.frombuffer(acc, dtype=np.uint8)
                acc.clear()
        finally:
            ex.shutdown(wait=False)

        if not offsets:
            return False

        f.create_dataset("offsets", data=np.array(offsets, dtype=np.int64), compression="gzip", compression_opts=4)
        f.create_dataset("lengths", data=np.array(lengths, dtype=np.uint32), compression="gzip", compression_opts=4)
        f.create_dataset("filenames", data=np.array(names, dtype=h5py.string_dtype()), compression="gzip", compression_opts=4)

    return True

def main():
    try:
        slides = [name for name in os.listdir(BASE_IN_SMB) if os.path.isdir(os.path.join(BASE_IN_SMB, name))]
    except Exception as e:
        print(f"Failed to list input directory: {e}")
        return

    print(f"Found {len(slides)} slides on Input NAS.")

    # Ensure output directory exists
    try:
        os.makedirs(BASE_OUT_SMB, exist_ok=True)
    except Exception:
        pass

    for name in tqdm(slides, desc="Slides", position=0, dynamic_ncols=True, mininterval=1.0):
        remote_out_file = os.path.join(BASE_OUT_SMB, f"{name}.h5")
        
        # Check if already exists on output NAS
        if os.path.exists(remote_out_file):
            tqdm.write(f"  Skipping {name} (already exists on output NAS)")
            continue

        tqdm.write(f"  Processing {name} ...")
        slide_path = os.path.join(BASE_IN_SMB, name)

        # Build HDF5 locally in a temporary directory to maximize I/O speed
        with tempfile.TemporaryDirectory() as temp_dir:
            local_h5 = os.path.join(temp_dir, f"{name}.h5")
            
            if not process_slide(slide_path, local_h5):
                tqdm.write(f"  No valid patches processed in {name}, skipped.")
                continue

            # Upload completed .h5 file to Output NAS
            try:
                tqdm.write(f"  Uploading {name}.h5 to Output NAS...")
                h5_size = os.path.getsize(local_h5)
                with open(local_h5, "rb") as f_src:
                    with open(remote_out_file, "wb") as f_dst:
                        while chunk := f_src.read(4 * 1024 * 1024): # 4MB chunks
                            f_dst.write(chunk)
            except Exception as e:
                tqdm.write(f"  [!] Failed to upload {name}.h5: {e}")

    print(f"\nDone processing all slides.")

if __name__ == "__main__":
    main()