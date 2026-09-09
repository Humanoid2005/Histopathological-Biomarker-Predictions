import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import PATCHES_PATH, EMBEDDINGS_PATH

class SlidePatchDataset(Dataset):
    """Custom Dataset to load patches for a single slide."""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # Open image and ensure 3 channels (RGB)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return image and just the filename (e.g., 'patch_level0_x1024_y2048.png')
        return image, os.path.basename(img_path)

def load_lunit_resnet50(weights_path, device):
    """Loads the SSL ResNet50 model and removes the classification head."""
    # Initialize a base ResNet50
    model = models.resnet50(weights=None)
    
    # Remove the fully connected layer to output raw embeddings (2048 dims)
    model.fc = nn.Identity()
    
    # Load the SSL weights
    state_dict = torch.load(weights_path, map_location=device)
    
    # SSL checkpoints often have prefixes from distributed training
    clean_state_dict = {}
    for k, v in state_dict.items():
        # Remove common SSL prefixes if they exist
        clean_key = k.replace("module.", "").replace("backbone.", "")
        clean_state_dict[clean_key] = v
        
    # strict=False allows us to safely ignore the missing 'fc' layer weights
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(device)
    model.eval() # Set to evaluation mode
    
    return model

def generate_embeddings(BASE_DIR_PATH, OUTPUT_DIR_PATH, model_weights_path, batch_size=256, num_workers=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)
    
    model = load_lunit_resnet50(model_weights_path, device)
    
    # Added Resize(224) to optimize 256x256 patches for the model's expected input
    transform = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get all slide directories (e.g., 'IN Brain-0002', 'IN Brain-0004(a)', etc.)
    slide_dirs = [d for d in os.listdir(BASE_DIR_PATH) if os.path.isdir(os.path.join(BASE_DIR_PATH, d))]
    print(f"Found {len(slide_dirs)} slide directories to process.")
    
    # Outer progress bar for slides
    for slide_name in tqdm(slide_dirs, desc="Total Progress"):
        slide_dir_path = os.path.join(BASE_DIR_PATH, slide_name)
        output_file_path = os.path.join(OUTPUT_DIR_PATH, f"{slide_name}.pth")
        
        # Fault tolerance: Skip if we already generated embeddings for this slide
        if os.path.exists(output_file_path):
            continue

        # Adjust the extension if your patches are saved as .jpg instead of .png
        patch_paths = glob.glob(os.path.join(slide_dir_path, "*.png"))
        
        if not patch_paths:
            print(f"\nWarning: No .png patches found in {slide_name}")
            continue
            
        dataset = SlidePatchDataset(patch_paths, transform=transform)
        
        # DataLoader handles multi-core CPU loading while GPU processes the current batch
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True # Speeds up CPU to GPU transfers
        )
        
        slide_embeddings = []
        slide_patch_names = []
        
        # Inner progress bar for batches within a slide
        with torch.no_grad():
            for images, filenames in tqdm(dataloader, desc=f"Embedding {slide_name}", leave=False):
                images = images.to(device, non_blocking=True)
                
                # Forward pass
                embeddings = model(images)
                
                # Move back to CPU to prevent VRAM overflow over 100k patches
                slide_embeddings.append(embeddings.cpu())
                slide_patch_names.extend(filenames)
        
        # Concatenate into a single tensor of shape [Num_Patches, 2048]
        slide_embeddings = torch.cat(slide_embeddings, dim=0)
        
        # Save as a dictionary to map embeddings back to their spatial coordinates
        torch.save({
            "embeddings": slide_embeddings,
            "patch_names": slide_patch_names
        }, output_file_path)

if __name__ == "__main__":
    base_folder = PATCHES_PATH 
    output_folder = EMBEDDINGS_PATH
    model_path = "../models/bt_rn50_ep200.torch"
    
    # Execute
    generate_embeddings(
        BASE_DIR_PATH=base_folder,
        OUTPUT_DIR_PATH=output_folder,
        model_weights_path=model_path,
        batch_size=256, 
        num_workers=8 
    )