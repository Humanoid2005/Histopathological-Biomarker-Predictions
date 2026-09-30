import torch
import torch.nn as nn
import gigapath.slide_encoder as slide_encoder
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

class GigapathFlashClassifier(nn.Module):
    def __init__(self, num_classes=1, model_path="slide_encoder.pth"):
        super().__init__()
        # Initialize the LongNet slide encoder for the Flash variant (384 input dimensions)
        self.slide_enc = slide_encoder.create_model(
            pretrained=model_path, 
            model_arch="gigapath_slide_enc12l384d",
            in_chans=384, 
            drop_path_rate=0.0
        )
        
        # Freeze most of the slide encoder, but unfreeze the last 2 layers and final norm
        for name, param in self.slide_enc.named_parameters():
            if "layers.10" in name or "layers.11" in name or "norm" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        # Add a classification head
        self.head = nn.Linear(384, num_classes)

    def forward(self, x, coords):
        # x shape: [B, N, D]
        # coords shape: [B, N, 2]
        outcomes = self.slide_enc(x, coords)
        
        # slide_enc returns a list of outcomes, we take the last one
        slide_level_embedding = outcomes[-1]
        
        logits = self.head(slide_level_embedding)
        return logits

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
