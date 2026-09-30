import os
import glob
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
import timm
import torch.nn as nn
from timm.layers import SwiGLUPacked

from config import PATCHES_PATH, EMBEDDINGS_PATH, MODEL_PATH

class SlidePatchDataset(Dataset):
    """Custom Dataset to load patches for a single slide."""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        import time
        import io
        max_retries = 5
        for attempt in range(max_retries):
            try:
                # Read entire file into memory instantly to release the NAS file lock.
                # This stops gvfs from deadlocking when using num_workers > 0
                with open(img_path, 'rb') as f:
                    img_bytes = f.read()
                
                image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\nWarning: Could not read {img_path} from NAS after {max_retries} attempts. Using blank patch.")
                    image = Image.new('RGB', (256, 256), color='black')
                else:
                    time.sleep(1.0 + (attempt * 0.5)) # Backoff to let NAS recover
        
        if self.transform:
            image = self.transform(image)
            
        return image, os.path.basename(img_path)

def load_gigapath_flash(weights_path, device):
    """Loads the ViT-S model from a locally downloaded .bin file."""
    
    # Create the bare architecture (ViT-Small, patch size 16, 224 resolution)
    # num_classes=0 removes the classification head to output raw embeddings
    model = timm.create_model(
        "vit_small_patch16_224", 
        pretrained=False, 
        num_classes=0, 
        act_layer=nn.SiLU,
        mlp_layer=SwiGLUPacked,
        mlp_ratio=16/3,
        init_values=1e-5,
        checkpoint_path=weights_path
    )
    
    model.to(device)
    model.eval() 
    return model

def generate_embeddings(BASE_DIR_PATH, OUTPUT_DIR_PATH, weights_path, batch_size=256, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    
    model = load_gigapath_flash(weights_path, device)
    
    # GigaPath-Flash expects 224x224 inputs to produce the 384-dim embeddings
    transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    slide_dirs = [d for d in os.listdir(BASE_DIR_PATH) if os.path.isdir(os.path.join(BASE_DIR_PATH, d))]
    print(f"Found {len(slide_dirs)} slide directories to process.")
    
    for slide_name in tqdm(slide_dirs, desc="Total Progress"):
        slide_dir_path = os.path.join(BASE_DIR_PATH, slide_name)
        output_file_path = os.path.join(OUTPUT_DIR_PATH, f"{slide_name}.pth")
        
        if os.path.exists(output_file_path):
            continue

        patch_paths = glob.glob(os.path.join(slide_dir_path, "*.png"))
        
        if not patch_paths:
            print(f"\nWarning: No .png patches found in {slide_name}")
            continue
            
        dataset = SlidePatchDataset(patch_paths, transform=transform)
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True 
        )
        
        slide_embeddings = []
        slide_patch_names = []
        
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc=f"Embedding {slide_name}", leave=False):
                images = images.to(device, non_blocking=True)
                
                # The timm ViT model directly outputs the pooled embedding sequence [Batch, 384]
                embeddings = model(images)
                
                slide_embeddings.append(embeddings.cpu())
                slide_patch_names.extend(filenames)
        
        # Concatenate into a single tensor of shape [Num_Patches, 384]
        slide_embeddings = torch.cat(slide_embeddings, dim=0)
        
        torch.save({
            "embeddings": slide_embeddings,
            "patch_names": slide_patch_names
        }, output_file_path)

if __name__ == "__main__":
    base_folder = PATCHES_PATH 
    output_folder = EMBEDDINGS_PATH
    tile_encoder_path = os.path.join(MODEL_PATH, "tile_encoder.pth")
    generate_embeddings(
        BASE_DIR_PATH=base_folder,
        OUTPUT_DIR_PATH=output_folder,
        weights_path=tile_encoder_path,
        batch_size=256, 
        num_workers=4 
    )