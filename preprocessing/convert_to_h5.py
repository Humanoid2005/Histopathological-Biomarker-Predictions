import os
import glob
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from config import PATCHES_PATH

def create_hdf5_for_slide(slide_dir, output_file, patch_size=256):
    """
    Converts a folder of patch images for a single slide into an HDF5 file.
    """
    # Grab all png and jpg images
    image_paths = glob.glob(os.path.join(slide_dir, "*.png"))
    image_paths.extend(glob.glob(os.path.join(slide_dir, "*.jpg")))
    image_paths.extend(glob.glob(os.path.join(slide_dir, "*.jpeg")))
    
    num_images = len(image_paths)
    if num_images == 0:
        return False
        
    with h5py.File(output_file, 'w') as h5_file:
        # Image dataset: optimized with chunking for reading one patch at a time
        images_ds = h5_file.create_dataset(
            name="images",
            shape=(num_images, patch_size, patch_size, 3),
            dtype=np.uint8,
            chunks=(1, patch_size, patch_size, 3), 
            compression="gzip",
            compression_opts=4
        )
        
        # Metadata dataset: stores the original filenames as UTF-8 strings
        string_dt = h5py.string_dtype(encoding='utf-8')
        filenames_ds = h5_file.create_dataset(
            name="filenames",
            shape=(num_images,),
            dtype=string_dt
        )

        # Iterate and write data
        for idx, path in enumerate(tqdm(image_paths, desc=f"Writing {os.path.basename(slide_dir)}", leave=False)):
            try:
                img = Image.open(path).convert('RGB')
                
                # Resize if necessary
                if img.size != (patch_size, patch_size):
                    img = img.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
                
                # Save to HDF5
                images_ds[idx] = np.array(img)
                filenames_ds[idx] = os.path.basename(path)
                
            except Exception as e:
                print(f"\nError processing {path}: {e}")
                
    return True

def main():
    base_dir = PATCHES_PATH
    # Create an output directory for the H5 files alongside the patches folder
    output_base_dir = os.path.join(os.path.dirname(base_dir), "patches_h5")
    os.makedirs(output_base_dir, exist_ok=True)
    
    slide_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Found {len(slide_dirs)} slide directories in {base_dir}")
    
    for slide_name in tqdm(slide_dirs, desc="Processing slides"):
        slide_dir = os.path.join(base_dir, slide_name)
        output_file = os.path.join(output_base_dir, f"{slide_name}.h5")
        
        # Fault tolerance: Skip if already processed
        if os.path.exists(output_file):
            continue 
            
        success = create_hdf5_for_slide(slide_dir, output_file)
        if not success:
            print(f"Skipped {slide_name} (no patches found).")
            
    print(f"\nFinished! HDF5 files saved to: {output_base_dir}")

if __name__ == "__main__":
    main()
