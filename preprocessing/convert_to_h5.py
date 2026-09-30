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

import smbclient

# --- Configuration ---
IN_NAS_IP = "172.16.201.2"
IN_SHARE = "prof-sushree"
IN_USER = "prof-sushree"
IN_PASS = "5Qmm*P"

OUT_NAS_IP = "172.16.202.70"
OUT_SHARE = "home"
OUT_USER = "ivanbh"
OUT_PASS = "i!DT7zDG"

smbclient.ClientConfig(session_timeout=36000)

smbclient.register_session(IN_NAS_IP, username=IN_USER, password=IN_PASS)
smbclient.register_session(OUT_NAS_IP, username=OUT_USER, password=OUT_PASS)

BASE_IN_SMB = rf"\\{IN_NAS_IP}\{IN_SHARE}\sriram-srikanth\patches"
BASE_OUT_SMB = rf"\\{OUT_NAS_IP}\{OUT_SHARE}\sriram-srikanth\patches"

WRITE_CHUNK = 64 * 1024 * 1024
EXTS = (".png", ".jpg", ".jpeg")
READ_WORKERS = 8
IN_FLIGHT = 16

def _read_smb(path: str) -> bytes:
    with smbclient.open_file(path, mode="rb") as f:
        return f.read()

def safe_scandir_smb(slide_dir):
    entries = []
    try:
        with smbclient.scandir(slide_dir) as it:
            for e in it:
                if e.is_file() and e.name.lower().endswith(EXTS):
                    entries.append((e.path, e.name))
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
        slides = [e.name for e in smbclient.scandir(BASE_IN_SMB) if e.is_dir()]
    except Exception as e:
        print(f"Failed to list input directory: {e}")
        return

    print(f"Found {len(slides)} slides on Input NAS.")

    try:
        smbclient.makedirs(BASE_OUT_SMB, exist_ok=True)
    except Exception:
        pass

    for name in tqdm(slides, desc="Slides", position=0, dynamic_ncols=True, mininterval=1.0):
        remote_out_file = rf"{BASE_OUT_SMB}\{name}.h5"
        
        try:
            if smbclient.stat(remote_out_file):
                tqdm.write(f"  Skipping {name} (already exists on output NAS)")
                continue
        except Exception:
            pass

        tqdm.write(f"  Processing {name} ...")
        slide_path = rf"{BASE_IN_SMB}\{name}"

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
                    with smbclient.open_file(remote_out_file, mode="wb") as f_dst:
                        with tqdm(total=h5_size, unit="B", unit_scale=True, desc=f"Uploading {name}.h5") as pbar:
                            while chunk := f_src.read(4 * 1024 * 1024): # 4MB chunks
                                f_dst.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                tqdm.write(f"  [!] Failed to upload {name}.h5: {e}")

    print(f"\nDone processing all slides.")

if __name__ == "__main__":
    main()