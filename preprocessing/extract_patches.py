import argparse
import csv
import os
import sys
from pathlib import Path
import concurrent.futures

import cv2
import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_WSI_SLIDES_PATH, PATCHES_PATH

class PatchExtractor:
    @staticmethod
    def create_tissue_mask(slide, downsample_factor=32):
        """Creates a binary mask of the tissue using a downsampled thumbnail."""
        width, height = slide.dimensions
        thumb_w = width // downsample_factor
        thumb_h = height // downsample_factor
        
        # Get thumbnail directly from openslide
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h))
        thumb_np = np.array(thumbnail)
        
        # Convert to HSV and apply Otsu thresholding on Saturation channel
        hsv = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2HSV)
        _, saturation, _ = cv2.split(hsv)
        
        _, mask = cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, downsample_factor

    @staticmethod
    def extract_and_save_patch(slide, x, y, patch_size, save_path):
        """Reads patch concurrently, converts directly from RGBA to BGR, and saves."""
        try:
            patch = slide.read_region((x, y), 0, (patch_size, patch_size))
            patch_np = np.array(patch)
            bgr_patch = cv2.cvtColor(patch_np, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(save_path, bgr_patch)
            patch.close()
            return True
        except Exception:
            return False

    @staticmethod
    def extract(tiff_dir_path, output_dir_path, patch_size, cloud_mode=True):
        os.makedirs(output_dir_path, exist_ok=True)

        log_path = os.path.join(output_dir_path, "patch_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as log_file:
            csv.writer(log_file).writerow(["filename", "num_patch"])

        filenames = sorted(os.listdir(tiff_dir_path))
        if not cloud_mode:
            filenames = tqdm(filenames, desc="Processing slides")

        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                tiff_path = os.path.join(tiff_dir_path, filename)
                saved_count, slide_name = PatchExtractor.extract_patches(tiff_path, output_dir_path, patch_size, cloud_mode)
                if slide_name is not None:
                    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
                        csv.writer(log_file).writerow([filename, saved_count])

    @staticmethod
    def extract_patches(tiff_path, output_base_dir, patch_size=256, cloud_mode=True):
        slide_name = os.path.splitext(os.path.basename(tiff_path))[0]
        output_dir = os.path.join(output_base_dir, slide_name)
        os.makedirs(output_dir, exist_ok=True)
        
        slide = None
        try:
            slide = openslide.OpenSlide(tiff_path)
            width, height = slide.dimensions
            
            # 1. Generate Tissue Mask
            mask, downsample_factor = PatchExtractor.create_tissue_mask(slide)
            
            # 2. Pre-calculate all valid coordinates using fast numpy checks
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
                    # 3. Only extract if tissue is present (threshold 0.5 or 50%)
                    if np.count_nonzero(mask_region) / (mask_region.size + 1e-6) >= 0.5:
                        valid_coords.append((x, y))

            saved_count = 0
            # Maximize thread workers for concurrent IO (read from slide & write to disk)
            max_workers = min(32, (os.cpu_count() or 4) * 2)
            
            # Execute both extraction and saving in the thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for x, y in valid_coords:
                    save_path = os.path.join(output_dir, f"{x}_{y}.png")
                    futures.append(
                        executor.submit(PatchExtractor.extract_and_save_patch, slide, x, y, patch_size, save_path)
                    )
                
                # Track progress of actual patch extraction/saving
                progress = concurrent.futures.as_completed(futures)
                if not cloud_mode:
                    progress = tqdm(progress, total=len(futures), desc=f"Extracting patches from {slide_name}")
                    
                for future in progress:
                    if future.result():
                        saved_count += 1
            
            return saved_count, slide_name

        except Exception as e:
            if not cloud_mode:
                print(f"Error loading {tiff_path}: {e}")
            return 0, None

        finally:
            if slide is not None:
                slide.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract patches from WSI slides")
    parser.add_argument(
        "--cloud-mode",
        type=lambda value: value.lower() == "true",
        default=True,
        help="Set to true to disable tqdm and enable logging (default: true)",
    )
    args = parser.parse_args()

    PatchExtractor.extract(
        RAW_WSI_SLIDES_PATH,
        PATCHES_PATH,
        patch_size=256,
        cloud_mode=args.cloud_mode,
    )