import torch
import os
import json
import numpy as np
from tqdm import tqdm

class Visualiser:
    def __init__(self, model, device, metrics_path, extract_coordinates_fn, patch_size=256):
        self.model = model
        self.device = device
        self.metrics_path = metrics_path
        self.extract_coordinates_fn = extract_coordinates_fn
        self.patch_size = patch_size

    def extract_and_visualise_rois(self, dataloader):
        """Method 1: Gradient-Based Attribution for identifying important patches."""
        self.model.eval()
        
        output_dir = os.path.join(self.metrics_path, "qupath_rois")
        os.makedirs(output_dir, exist_ok=True)
        
        progress_bar = tqdm(dataloader, desc="Extracting ROIs")
        for step, (data, label) in enumerate(progress_bar):
            label = label.to(self.device).float()
            
            if isinstance(data, dict):
                embeddings = data["embeddings"].to(self.device)
                patch_names = data.get("patch_names", [f"patch_{i}" for i in range(embeddings.shape[1])])
                if isinstance(patch_names, list) and len(patch_names) > 0 and isinstance(patch_names[0], (tuple, list)):
                    names = [p[0] for p in patch_names]
                else:
                    names = patch_names
            else:
                embeddings = data.to(self.device)
                names = [f"patch_{i}" for i in range(embeddings.shape[1])]
                
            coordinates = self.extract_coordinates_fn(data)
            
            # Enable Gradients on Inputs
            embeddings.requires_grad_()
            
            # Forward pass
            logits = self.model(embeddings, coordinates)
            
            # Backward pass on the logit
            logits.backward(torch.ones_like(logits))
            
            # Calculate Patch Importance (L2 norm of gradients)
            importance_scores = torch.norm(embeddings.grad, dim=-1).squeeze(0) # Shape: [NumPatches]
            
            coords_np = coordinates.squeeze(0).cpu().numpy()
            scores_np = importance_scores.cpu().numpy()
            
            try:
                # Extract slide name from the first patch name (e.g., slide_id_x_y.png -> slide_id)
                slide_name = "_".join(names[0].split("_")[:-2]) if "_" in names[0] else f"slide_{step}"
            except Exception:
                slide_name = f"slide_{step}"
            if not slide_name: slide_name = f"slide_{step}"
            
            # Filter for top 5% of patches to avoid cluttering QuPath with thousands of annotations
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