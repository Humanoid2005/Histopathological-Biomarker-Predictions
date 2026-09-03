import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import openslide
from PIL import Image
from tqdm import tqdm

# Allow direct execution from the preprocessing directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import RAW_WSI_SLIDES_PATH, PATCHES_PATH

class PatchExtractor:
    @staticmethod
    def extract(tiff_dir_path,output_dir_path,patch_size,cloud_mode=True):
        """
            Extracts patches from all the WSI images in the specified directory.

        Args:
            tiff_dir_path (str): Path to the directory containing WSI images.
            output_dir_path (str): Path to the directory where extracted patches will be saved.
            patch_size (int): Size of the square patches to extract.
        """
        # Ensure output directory exists
        os.makedirs(output_dir_path, exist_ok=True)

        log_path = os.path.join(output_dir_path, "patch_log.csv")
        if cloud_mode:
            with open(log_path, "w", newline="", encoding="utf-8") as log_file:
                csv.writer(log_file).writerow(["filename", "num_patch"])

        # Iterate over all files in the input directory
        filenames = os.listdir(tiff_dir_path)
        if not cloud_mode:
            filenames = tqdm(filenames, desc="Processing slides")

        for filename in filenames:
            if filename.lower().endswith(('.tif', '.tiff')):
                tiff_path = os.path.join(tiff_dir_path, filename)
                saved_count,slide_name = PatchExtractor.extract_patches(tiff_path, output_dir_path, patch_size, cloud_mode)
                if cloud_mode:
                    with open(log_path, "a", newline="", encoding="utf-8") as log_file:
                        csv.writer(log_file).writerow([filename, saved_count])
    @staticmethod
    def extract_patches(tiff_path, output_base_dir, patch_size=256, cloud_mode=True):
        """
            Since the images have already been normalised and Otsu thresholding has been applied, 
            we can directly extract patches from the WSI.

        Args:
            tiff_path (_type_): _description_
            output_base_dir (_type_): _description_
            patch_size (int, optional): _description_. Defaults to 256.
        """
        
        # 1. Setup folder structure
        slide_name = os.path.splitext(os.path.basename(tiff_path))[0]
        output_dir = os.path.join(output_base_dir, slide_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Load Whole Slide Image
        try:
            slide = openslide.OpenSlide(tiff_path)
        except Exception as e:
            print(f"Error loading {tiff_path}: {e}")
            return
            
        width, height = slide.dimensions # Dimensions at max resolution (level 0)
        saved_count = 0
        
        # 3. Sliding window across the coordinate grid
        y_positions = range(0, height, patch_size)
        if not cloud_mode:
            y_positions = tqdm(y_positions, desc=f"Extracting patches from {slide_name}")

        for y in y_positions:
            for x in range(0, width, patch_size):
                
                # Extract the 256x256 patch at level 0
                patch = slide.read_region((x, y), 0, (patch_size, patch_size))
                
                # Convert RGBA to RGB
                patch_rgb = patch.convert('RGB')
                patch_array = np.array(patch_rgb)
                
                # 4. Background Filtering
                # If the patch is overwhelmingly white/bright, skip it.
                # 240 out of 255 is a standard threshold for empty glass.
                if np.mean(patch_array) > 240:
                    continue
                
                # 5. Save valid patch
                filename = f"{x}_{y}.png"
                save_path = os.path.join(output_dir, filename)
                patch_rgb.save(save_path, format="PNG")
                saved_count += 1
            
        return saved_count,slide_name
    
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