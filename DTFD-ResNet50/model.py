import torch
import torch.nn as nn
import torch.nn.functional as F

class Bag:
    def __init__(self):
        self.patches = []

    def add_patch(self,index,coordinate,embedding):
        self.patches.append([index,coordinate,embedding])

    def get_patches(self):
        return self.patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self,idx):
        return self.patches[idx][0],self.patches[idx][1],self.patches[idx][2]

class AttentionModule(nn.Module):
    def __init__(self, in_features, hidden_dim=256):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A_v = self.attention_v(x)
        A_u = self.attention_u(x)
        A = self.attention_weights(A_v * A_u)
        A = torch.transpose(A, 1, 0) # [1, N]
        A = F.softmax(A, dim=1) # [1, N]
        return A

class TierMIL(nn.Module):
    def __init__(self, in_features, projected_dim=512, hidden_dim=256, out_classes=1, dropout=0.25):
        super().__init__()
        
        # 1. Feature Projection (Adds deep non-linearity)
        self.projector = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, projected_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Attention Module
        self.attention = AttentionModule(projected_dim, hidden_dim)
        
        # 3. Classifier (Multi-layer instead of single linear)
        self.classifier = nn.Sequential(
            nn.Linear(projected_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_classes)
        )

    def forward(self, x):
        # x is [N, in_features]
        h = self.projector(x) # [N, projected_dim]
        A = self.attention(h) # [1, N]
        M = torch.mm(A, h)    # [1, projected_dim]  -> Aggregated Feature (AFS)
        logits = self.classifier(M) # [1, out_classes]
        return logits, M, A

class DTFDModel(nn.Module):
    def __init__(self, in_features, num_bags, out_classes=1, hidden_dim=256, projected_dim=512, dropout=0.25):
        super().__init__()
        self.in_features = in_features
        # num_bags here is M, the number of pseudo bags. 
        self.num_pseudo_bags = num_bags 
        
        # Double-Tier Architecture
        # Tier 1 takes original features (2048) and projects to projected_dim
        self.tier1 = TierMIL(in_features, projected_dim, hidden_dim, out_classes, dropout)
        # Tier 2 takes the aggregated features from Tier 1, and projects again
        self.tier2 = TierMIL(projected_dim, projected_dim, hidden_dim, out_classes, dropout)

    def forward(self, X):
        """
        X is expected to be a tensor of shape [Total_Patches, in_features]
        If it's a dict containing 'embeddings', we extract it first.
        """
        patch_names = None
        if isinstance(X, dict) and "embeddings" in X:
            embeddings = X["embeddings"]
            patch_names = X.get("patch_names", None)
        else:
            embeddings = X

        # Squeeze batch dimension if present (DataLoader with batch_size=1 adds a dimension)
        if embeddings.dim() == 3 and embeddings.size(0) == 1:
            embeddings = embeddings.squeeze(0)
            
        if patch_names is not None and isinstance(patch_names, (list, tuple)):
            # DataLoader adds a batch dimension to lists as well (tuple of lists/tuples)
            if len(patch_names) == 1 and isinstance(patch_names[0], (list, tuple)):
                patch_names = patch_names[0]
            
        N = embeddings.size(0)
        M = self.num_pseudo_bags
        
        # If N < M, we just use N pseudo bags
        if N < M:
            M = N
            
        # Attempt spatial clustering
        pseudo_bags = []
        spatial_success = False
        
        if patch_names is not None and len(patch_names) == N:
            try:
                coords = []
                for name in patch_names:
                    # name format from extract_patches.py: "X_Y.png"
                    parts = name.replace('.png', '').replace('.jpg', '').split('_')
                    if len(parts) >= 2:
                        coords.append([float(parts[0]), float(parts[1])])
                    else:
                        coords.append([0.0, 0.0])
                
                from sklearn.cluster import KMeans
                import numpy as np
                
                coords_np = np.array(coords)
                kmeans = KMeans(n_clusters=M, n_init=1, random_state=42)
                labels = kmeans.fit_predict(coords_np)
                
                for i in range(M):
                    bag_indices = torch.tensor(np.where(labels == i)[0], dtype=torch.long, device=embeddings.device)
                    if len(bag_indices) > 0:
                        pseudo_bags.append(embeddings[bag_indices])
                        
                # Ensure we got exactly M bags with no empty clusters
                if len(pseudo_bags) == M:
                    spatial_success = True
            except Exception as e:
                pass # Fallback to random partition
                
        if not spatial_success:
            pseudo_bags = []
            # Shuffle indices for random partition fallback
            indices = torch.randperm(N, device=embeddings.device)
            chunk_size = N // M
            remainder = N % M
            
            start = 0
            for i in range(M):
                end = start + chunk_size + (1 if i < remainder else 0)
                bag_indices = indices[start:end]
                pseudo_bags.append(embeddings[bag_indices])
                start = end
            
        tier1_logits = []
        distilled_features = []
        
        # Process Tier 1
        for bag_embeddings in pseudo_bags:
            # bag_embeddings: [K, in_features]
            logits, M_feat, A = self.tier1(bag_embeddings)
            tier1_logits.append(logits)
            # AFS: Aggregated feature selection
            distilled_features.append(M_feat.squeeze(0))
            
        tier1_logits = torch.cat(tier1_logits, dim=0) # [M, out_classes]
        distilled_features = torch.stack(distilled_features, dim=0) # [M, in_features]
        
        # Process Tier 2
        tier2_logits, _, tier2_A = self.tier2(distilled_features) # [1, out_classes]
        
        return tier1_logits, tier2_logits, tier2_A

    def load_model(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))

    def save_model(self, path):
        torch.save(self.state_dict(), path)
