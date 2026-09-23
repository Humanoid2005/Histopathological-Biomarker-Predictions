import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from model import DTFDModel
import numpy as np
from sklearn.metrics import roc_curve

class BioMarkerPredictor:
    def __init__(self, device,num_bags,metrics_path):
        self.device = device
        self.metrics_path = metrics_path
        # Using num_bags=5 as per DTFD paper for CAMELYON-16. 
        # With 100k patches, M=5 gives 20k patches per pseudo-bag which fits perfectly in GPU RAM
        self.model = DTFDModel(in_features=2048, num_bags=num_bags, out_classes=1)
        self.model.to(device)
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def fit(self, train_dataloader, val_dataloader=None, epochs=10, lr=0.001, weight_decay=1e-5):
        # Update optimizer params if provided differently
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
            
        # Mixed Precision Scaler for faster training and less memory footprint
        scaler = torch.amp.GradScaler('cuda')
        
        # We accumulate gradients over 32 steps to simulate batch_size=32, 
        # while keeping real batch_size=1 to avoid variable-length tensor collation errors.
        accumulation_steps = 32

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for step, (data, label) in enumerate(progress_bar):
                # Data might be a tuple or dict, handle appropriately
                label = label.to(self.device).float()
                if isinstance(data, dict):
                    embeddings = data["embeddings"].to(self.device)
                else:
                    embeddings = data.to(self.device)
                
                # Automatic Mixed Precision
                with torch.amp.autocast('cuda'):
                    tier1_logits, tier2_logits, _ = self.model(embeddings)
                    
                    # Tier 1 Loss: all pseudo bags inherit the slide's parent label
                    tier1_labels = label.expand(tier1_logits.size(0), 1)
                    loss1 = self.loss_fn(tier1_logits, tier1_labels)
                    
                    # Tier 2 Loss: parent bag prediction
                    loss2 = self.loss_fn(tier2_logits, label.view(1, 1))
                    
                    loss = (loss1 + loss2) / accumulation_steps
                
                # Scaled Backward pass
                scaler.scale(loss).backward()
                train_loss += loss.item() * accumulation_steps
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    
                progress_bar.set_postfix({'loss': f"{train_loss / (step + 1):.4f}"})
                
            # Validation at end of epoch
            if val_dataloader is not None:
                self.evaluate(val_dataloader, epoch, epochs)

    def evaluate(self, dataloader, epoch, epochs):
        self.model.eval()
        all_probs = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for data, label in progress_bar:
                label = label.to(self.device).float()
                if isinstance(data, dict):
                    embeddings = data["embeddings"].to(self.device)
                else:
                    embeddings = data.to(self.device)
                
                with torch.amp.autocast('cuda'):
                    tier1_logits, tier2_logits, _ = self.model(embeddings)
                    loss = self.loss_fn(tier2_logits, label.view(1, 1))
                    val_loss += loss.item()
                    
                prob = torch.sigmoid(tier2_logits).item()
                all_probs.append(prob)
                all_labels.append(label.item())
                
                progress_bar.set_postfix({'val_loss': f"{val_loss / len(all_probs):.4f}"})
                
        # Calculate optimal threshold using ROC curve
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            # Youden's J statistic to find the best threshold
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        except ValueError:
            optimal_threshold = 0.5
        
        acc, auc, f1, precision, recall = self.save_metrics(all_labels, all_probs, optimal_threshold,epoch)
        print(f"--> Validation Metrics: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f} (Threshold: {optimal_threshold:.4f})")

    def predict(self, test_dataloader):
        print("Starting Prediction on Test Set...")
        self.evaluate(test_dataloader, epoch=0, epochs=1)

    def save_metrics(self, all_labels, all_probs, threshold,epoch):
        import os
        import matplotlib.pyplot as plt
        
        preds = [1 if p >= threshold else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        
        try:
            auc = roc_auc_score(all_labels, all_probs)
            fpr, tpr, _ = roc_curve(all_labels, all_probs)
        except ValueError:
            auc = 0.5
            fpr, tpr = [0, 1], [0, 1]
            
        f1 = f1_score(all_labels, preds, zero_division=0)
        precision = precision_score(all_labels, preds, zero_division=0)
        recall = recall_score(all_labels, preds, zero_division=0)
        
        # Save ROC Plot
        os.makedirs(self.metrics_path, exist_ok=True)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.metrics_path, f'{epoch}_roc_curve.png'))
        plt.close()
        
        with open(os.path.join(self.metrics_path, f'{epoch}_metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Optimal Threshold: {threshold:.4f}\n")
            
        return acc, auc, f1, precision, recall
        
    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
