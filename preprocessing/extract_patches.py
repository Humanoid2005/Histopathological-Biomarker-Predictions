# Limit native thread pools (BLAS etc.) *before* numpy/cv2 are imported,
# so this takes effect in all threads.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import signal
import sys
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import h5py
import openslide
from tqdm import tqdm

# Prevent OpenCV from spinning up its own threads inside our worker threads
cv2.setNumThreads(1)

import smbclient
import tempfile
import os

# --- Configuration ---
IN_NAS_IP = "172.16.201.2"
IN_SHARE = "prof-sushree"
IN_USER = "prof-sushree"
IN_PASS = "5Qmm*P"

OUT_NAS_IP = "172.16.202.70"
OUT_SHARE = "home"
OUT_USER = "ivanbh"
OUT_PASS = "i!DT7zDG"

# Crucial: Keep large file uploads/downloads alive
smbclient.ClientConfig(session_timeout=36000)

# Register SMB sessions for both servers
smbclient.register_session(IN_NAS_IP, username=IN_USER, password=IN_PASS)
smbclient.register_session(OUT_NAS_IP, username=OUT_USER, password=OUT_PASS)

# SMB UNC Paths
BASE_IN_SMB = rf"\\{IN_NAS_IP}\{IN_SHARE}\sriram-srikanth\images"
BASE_OUT_SMB = rf"\\{OUT_NAS_IP}\{OUT_SHARE}\sriram-srikanth\patches"

# Ensure output directory exists
try:
    smbclient.makedirs(BASE_OUT_SMB, exist_ok=True)
except Exception:
    pass


completed_slides = [
    "IN Brain-0002.tiff", "IN Brain-0004(a).tiff", "IN Brain-0004(b).tiff",
    "IN Brain-0007(a).tiff", "IN Brain-0007(b).tiff", "IN Brain-0008(a).tiff",
    "IN Brain-0008(b).tiff", "IN Brain-0012(a).tiff", "IN Brain-0012(b).tiff",
    "IN Brain-0013(a).tiff", "IN Brain-0013(b).tiff", "IN Brain-0015(a).tiff",
    "IN Brain-0015(b).tiff", "IN Brain-0020.tiff", "IN Brain-0021.tiff",
    "IN Brain-0024.tiff", "IN Brain-0025(a).tiff", "IN Brain-0025(b).tiff",
    "IN Brain-0029(a).tiff", "IN Brain-0029(b).tiff", "IN Brain-0029(c).tiff",
    "IN Brain-0030.tiff", "IN Brain-0031(a).tiff", "IN Brain-0031(b).tiff",
    "IN Brain-0031(c).tiff", "IN Brain-0031(d).tiff", "IN Brain-0034.tiff",
    "IN Brain-0035(a).tiff", "IN Brain-0035(b).tiff", "IN Brain-0036(a).tiff",
    "IN Brain-0036(b).tiff", "IN Brain-0038.tiff",'IN Brain-0050(a).tiff',
    'IN Brain-0050(b).tiff','IN Brain-0052(a).tiff','IN Brain-0052(b).tiff',
    'IN Brain-0052(c).tiff','IN Brain-0056.tiff','IN Brain-0057.tiff','IN Brain-0058(a).tiff',
    'IN Brain-0058(b).tiff','IN Brain-0059(a).tiff','IN Brain-0059(b).tiff','IN Brain-0059(c).tiff',
    'IN Brain-0060.tiff','IN Brain-0061(a).tiff','IN Brain-0061(b).tiff','IN Brain-0062(a).tiff',
    'IN Brain-0062(b).tiff','IN Brain-0063(a).tiff','IN Brain-0063(b).tiff','IN Brain-0063(c).tiff'
]


# in_personal_harddisk = [
#     "IN Brain-0039.tiff",
#     "IN Brain-0040(a).tiff", "IN Brain-0040(b).tiff", "IN Brain-0041(a).tiff",
#     "IN Brain-0041(b).tiff", "IN Brain-0042.tiff", "IN Brain-0044(a).tiff",
#     "IN Brain-0044(b).tiff", "IN Brain-0045.tiff", "IN Brain-0046.tiff",
#     "IN Brain-0047.tiff", "IN Brain-0048.tiff"
# ]

# completed_slides.extend(in_personal_harddisk)
# How many (x, y) coordinates get handed to the pool at once, instead of
# submitting every patch for a slide (can be 50k+) in one shot. Keeps the
# in-flight Future bookkeeping small and gives natural checkpoints.
CHUNK_SIZE = 2000

# How often (seconds) the main thread wakes up to check for Ctrl+C, even if
# nothing has finished yet. This is the actual mechanism that guarantees we
# never block forever on a stalled NAS write.
POLL_TIMEOUT = 2.0

# Thread-local storage guarantees each thread gets its own unique file handle to the TIFF.
# This prevents the libtiff library from locking up when multiple threads read at once.
thread_local = threading.local()

# Set by the SIGINT/SIGTERM handler below. Every wait loop checks this
# instead of letting a blocked worker thread hang the whole process.
stop_event = threading.Event()


def _request_stop(signum, frame):
    stop_event.set()


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)


def get_slide_for_thread(tiff_path):
    if not hasattr(thread_local, "slide"):
        thread_local.slide = openslide.OpenSlide(tiff_path)
    return thread_local.slide


def create_tissue_mask(slide, downsample_factor=32):
    width, height = slide.dimensions
    thumb_w = width // downsample_factor
    thumb_h = height // downsample_factor

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
    thumb_np = np.array(thumbnail)

    hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
    _, saturation, _ = cv2.split(hsv)

    _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask, downsample_factor


def _extract_one_patch(args):
    """Executes on a single thread. Returns (success_bool, result)."""
    tiff_path, x, y, patch_size = args
    name = f"{x}_{y}.png"

    try:
        # Fetch the thread-independent slide object
        slide = get_slide_for_thread(tiff_path)

        # Read the raw patch
        patch = slide.read_region((x, y), 0, (patch_size, patch_size))
        patch_np = np.array(patch)
        patch.close()

        # Convert RGBA (openslide) to BGR (opencv)
        bgr_patch = cv2.cvtColor(patch_np, cv2.COLOR_RGBA2BGR)

        # Use imencode with fast compression (1) to keep encode speeds high
        success, encoded = cv2.imencode('.png', bgr_patch, [int(cv2.IMWRITE_PNG_COMPRESSION), 1])

        if not success:
            return False, f"cv2.imencode failed at ({x}, {y})"
        return True, (name, encoded.tobytes())

    except Exception as e:
        return False, str(e)


def _chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _emergency_exit(log_file, slide_name, saved_count):
    """Write what we have to the CSV and kill the process immediately.

    Deliberately NOT sys.exit() / return here. If a worker thread is stuck
    on a slow or wedged NAS write, the normal shutdown path -- the thread
    pool's own cleanup, and the atexit hook that joins every pool thread on
    interpreter exit -- will wait for that thread, and Ctrl+C will look like
    it does nothing. os._exit() ends the process at the OS level and skips
    all of that, so a stuck thread can never block a real shutdown.
    """
    try:
        csv.writer(log_file).writerow([slide_name, saved_count, "interrupted"])
        log_file.flush()
        log_file.close()
    except Exception:
        pass
    print(f"\nInterrupted. Saved {saved_count} patches from {slide_name} before stopping.")
    os._exit(130)  # 128 + SIGINT


def process_slide(tiff_path, output_base_dir, patch_size, num_workers, log_file):
    slide_name = os.path.splitext(os.path.basename(tiff_path))[0]
    h5_path = os.path.join(output_base_dir, f"{slide_name}.h5")

    saved_count = 0
    try:
        # Open just once in the main thread to build the mask and coordinates
        with openslide.OpenSlide(tiff_path) as slide:
            width, height = slide.dimensions
            mask, downsample_factor = create_tissue_mask(slide)

        valid_coords = []
        for y in range(0, height - patch_size + 1, patch_size):
            for x in range(0, width - patch_size + 1, patch_size):
                mask_x = int(x / downsample_factor)
                mask_y = int(y / downsample_factor)
                mask_w = max(1, int(patch_size / downsample_factor))
                mask_h = max(1, int(patch_size / downsample_factor))

                if mask_y >= mask.shape[0] or mask_x >= mask.shape[1]:
                    continue

                mask_region = mask[mask_y : mask_y + mask_h, mask_x : mask_x + mask_w]
                if np.count_nonzero(mask_region) / (mask_region.size + 1e-6) >= 0.5:
                    valid_coords.append((x, y))

        total_valid = len(valid_coords)
        if total_valid == 0:
            return "ok", 0

        if stop_event.is_set():
            _emergency_exit(log_file, slide_name, saved_count)

        pbar = tqdm(total=total_valid, desc=slide_name, file=sys.stdout)
        executor = ThreadPoolExecutor(max_workers=num_workers)

        with h5py.File(h5_path, "w", libver="latest") as f:
            buf = f.create_dataset("image_bytes", shape=(0,), maxshape=(None,), 
                                   dtype=np.uint8, chunks=(4 * 1024 * 1024,))
            offsets = []
            lengths = []
            names = []
            
            acc = bytearray()
            current_offset = 0

            # Feed the pool in bounded batches rather than submitting every
            # patch at once. Each batch is polled with a timeout instead of
            # waiting on it indefinitely, so even if every worker is currently
            # stuck on a slow NAS write, we still wake up every POLL_TIMEOUT
            # seconds to check for Ctrl+C.
            for chunk in _chunked(valid_coords, CHUNK_SIZE):
                tasks = [(tiff_path, x, y, patch_size) for x, y in chunk]
                pending = {executor.submit(_extract_one_patch, t) for t in tasks}

                while pending:
                    done, pending = wait(pending, timeout=POLL_TIMEOUT, return_when=FIRST_COMPLETED)

                    for future in done:
                        success, result = future.result()
                        if success:
                            name, data = result
                            length = len(data)
                            if length > 0:
                                offsets.append(current_offset)
                                lengths.append(length)
                                names.append(name)
                                acc.extend(data)
                                current_offset += length
                                
                                if len(acc) >= 64 * 1024 * 1024:
                                    old_len = buf.shape[0]
                                    new_len = old_len + len(acc)
                                    buf.resize((new_len,))
                                    buf[old_len:new_len] = np.frombuffer(acc, dtype=np.uint8)
                                    acc.clear()
                                    
                            saved_count += 1
                        pbar.update(1)

                    if stop_event.is_set():
                        pbar.close()
                        executor.shutdown(wait=False)
                        _emergency_exit(log_file, slide_name, saved_count)

            if acc:
                old_len = buf.shape[0]
                new_len = old_len + len(acc)
                buf.resize((new_len,))
                buf[old_len:new_len] = np.frombuffer(acc, dtype=np.uint8)
                acc.clear()
                
            if offsets:
                f.create_dataset("offsets", data=np.array(offsets, dtype=np.int64), compression="gzip", compression_opts=4)
                f.create_dataset("lengths", data=np.array(lengths, dtype=np.uint32), compression="gzip", compression_opts=4)
                f.create_dataset("filenames", data=np.array(names, dtype=h5py.string_dtype()), compression="gzip", compression_opts=4)

        executor.shutdown(wait=True)
        pbar.close()
        return "ok", saved_count

    except Exception as e:
        print(f"[{slide_name}] FAILED: {e}", file=sys.stderr)
        return "error", saved_count


def extract(patch_size, num_workers):
    try:
        entries = smbclient.scandir(BASE_IN_SMB)
        filenames = sorted([e.name for e in entries if e.is_file() and e.name.lower().endswith((".tif", ".tiff"))])
    except Exception as e:
        print(f"Failed to list input directory via smbclient: {e}")
        return

    pending_slides = [f for f in filenames if f not in completed_slides]
    print(f"Found {len(pending_slides)} slides to process (skipping {len(completed_slides)} already completed).")

    for idx, filename in enumerate(pending_slides, start=1):
        if stop_event.is_set():
            break

        slide_name = os.path.splitext(filename)[0]
        remote_in_file = rf"{BASE_IN_SMB}\{filename}"
        remote_out_file = rf"{BASE_OUT_SMB}\{slide_name}.h5"

        # Check if already exists on output NAS
        try:
            if smbclient.stat(remote_out_file):
                print(f"[{idx}/{len(pending_slides)}] Skipping {filename} (H5 already exists on NAS)")
                continue
        except Exception:
            pass

        print(f"[{idx}/{len(pending_slides)}] Starting {filename}")

        with tempfile.TemporaryDirectory() as temp_dir:
            local_tiff = os.path.join(temp_dir, filename)
            
            # Download TIFF locally for faster openslide access
            try:
                file_size = smbclient.stat(remote_in_file).st_size
                with smbclient.open_file(remote_in_file, mode="rb") as f_src:
                    with open(local_tiff, "wb") as f_dst:
                        with tqdm(total=file_size, unit="B", unit_scale=True, desc=f"Downloading {filename}") as pbar:
                            while chunk := f_src.read(1024 * 1024):  # 1MB chunks
                                f_dst.write(chunk)
                                pbar.update(len(chunk))
            except Exception as e:
                print(f"  [!] Failed to download {filename}: {e}")
                continue

            # Process
            log_file = open(os.devnull, "w")
            status, saved_count = process_slide(local_tiff, temp_dir, patch_size, num_workers, log_file)
            log_file.close()

            print(f"  Finished extracting {filename} locally - status={status}, patches={saved_count}")

            if status == "ok" and saved_count > 0:
                local_h5 = os.path.join(temp_dir, f"{slide_name}.h5")
                if os.path.exists(local_h5):
                    try:
                        h5_size = os.path.getsize(local_h5)
                        with open(local_h5, "rb") as f_src:
                            with smbclient.open_file(remote_out_file, mode="wb") as f_dst:
                                with tqdm(total=h5_size, unit="B", unit_scale=True, desc=f"Uploading {slide_name}.h5") as pbar:
                                    while chunk := f_src.read(1024 * 1024): # 1MB chunks
                                        f_dst.write(chunk)
                                        pbar.update(len(chunk))
                        print(f"  Successfully uploaded {slide_name}.h5")
                    except Exception as e:
                        print(f"  [!] Failed to upload {slide_name}.h5: {e}")

    print("All slides processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI slides")
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel threads to use per slide (default: 16)",
    )
    args = parser.parse_args()

    try:
        extract(
            patch_size=256,
            num_workers=args.workers,
        )
    except KeyboardInterrupt:
        # Fallback only - the SIGINT handler + stop_event checks above should
        # already have exited via os._exit() before this is ever reached.
        os._exit(130)