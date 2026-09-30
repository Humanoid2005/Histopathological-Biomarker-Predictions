import torch
import os
import json
import numpy as np
from tqdm import tqdm
import re

class Visualiser:
    def __init__(self, model, device, metrics_path, patch_size=256):
        self.model = model
        self.device = device
        self.metrics_path = metrics_path
        self.patch_size = patch_size

    def extract_and_visualise_rois(self, dataloader):
        """Gradient-Based Attribution for identifying important patches in DTFD."""
        self.model.eval()
        
        output_dir = os.path.join(self.metrics_path, "qupath_rois")
        os.makedirs(output_dir, exist_ok=True)
        
        progress_bar = tqdm(dataloader, desc="Extracting ROIs")
        for step, (data, label) in enumerate(progress_bar):
            
            if isinstance(data, dict):
                embeddings = data["embeddings"].to(self.device)
                patch_names = data.get("patch_names", [])
                if isinstance(patch_names, list) and len(patch_names) > 0 and isinstance(patch_names[0], (tuple, list)):
                    names = [p[0] for p in patch_names]
                else:
                    names = patch_names
            else:
                embeddings = data.to(self.device)
                names = [f"patch_{i}" for i in range(embeddings.shape[1])]
                
            # Extract coordinates from names
            coords = []
            for name in names:
                if isinstance(name, str):
                    match = re.search(r'(\d+)[_,-](\d+)\.[a-zA-Z]+$', name)
                    if match:
                        coords.append([float(match.group(1)), float(match.group(2))])
                    else:
                        coords.append([0.0, 0.0])
                else:
                    coords.append([0.0, 0.0])
            coords_np = np.array(coords)
            
            # Enable Gradients on Inputs
            embeddings.requires_grad_()
            
            # Forward pass
            tier1_logits, tier2_logits, _ = self.model({"embeddings": embeddings, "patch_names": names})
            
            # Backward pass on the logit
            tier2_logits.backward(torch.ones_like(tier2_logits))
            
            # Calculate Patch Importance (L2 norm of gradients)
            importance_scores = torch.norm(embeddings.grad, dim=-1).squeeze(0) # Shape: [NumPatches]
            scores_np = importance_scores.cpu().numpy()
            
            if isinstance(data, dict) and "slide_name" in data:
                slide_name = data["slide_name"]
                if isinstance(slide_name, (list, tuple)):
                    slide_name = slide_name[0]
            else:
                slide_name = f"slide_{step}"
                
            # Filter for top 5% of patches
            k = max(1, int(0.05 * len(scores_np)))
            threshold_score = np.sort(scores_np)[-k]
            
            features = []
            for idx in range(len(scores_np)):
                score = float(scores_np[idx])
                if score < threshold_score:
                    continue
                    
                x, y = int(coords_np[idx][0]), int(coords_np[idx][1])
                
                # QuPath standard GeoJSON feature
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [x, y],
                                [x + self.patch_size, y],
                                [x + self.patch_size, y + self.patch_size],
                                [x, y + self.patch_size],
                                [x, y]
                            ]
                        ]
                    },
                    "properties": {
                        "objectType": "annotation",
                        "classification": {
                            "name": "High Importance ROI",
                            "color": [255, 0, 0] # Red
                        },
                        "measurements": [
                            {"name": "ImportanceScore", "value": score}
                        ]
                    }
                }
                features.append(feature)
                
            geojson = {
                "type": "FeatureCollection",
                "features": features
            }
            
            with open(os.path.join(output_dir, f"{slide_name}_rois.geojson"), "w") as f:
                json.dump(geojson, f)
            
            self.model.zero_grad()