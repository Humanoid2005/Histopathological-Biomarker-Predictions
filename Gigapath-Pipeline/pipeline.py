import os
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from model import GigapathFlashClassifier
import numpy as np
import re
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualisation import Visualiser

class BioMarkerPredictor:
    def __init__(self, device, metrics_path, slide_encoder_path="slide_encoder.pth"):
        self.device = device
        self.metrics_path = metrics_path
        
        # Initialize the Flash variant model
        self.model = GigapathFlashClassifier(num_classes=1, model_path=slide_encoder_path)
        self.model.to(device)
        
        # Setup optimizer and loss
        # Since slide_enc is frozen, we only optimize the head
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _extract_coordinates(self, data):
        """Extract spatial coordinates from patch names. 
        Returns coordinates tensor of shape [Batch, NumPatches, 2]."""
        
        # Default empty coords if not a dictionary with patch names
        if not isinstance(data, dict) or "patch_names" not in data:
            if isinstance(data, dict):
                embeddings = data["embeddings"]
            else:
                embeddings = data
            return torch.zeros((embeddings.shape[0], embeddings.shape[1], 2), dtype=torch.float32, device=self.device)
            
        patch_names = data["patch_names"]
        # Handle default_collate behavior where batch_size=1 turns list into list of tuples
        names = [p[0] if isinstance(p, (tuple, list)) else p for p in patch_names]
        
        coords = []
        for name in names:
            if isinstance(name, str):
                match = re.search(r'(\d+)[_,-](\d+)\.[a-zA-Z]+$', name)
                if match:
                    coords.append([int(match.group(1)), int(match.group(2))])
                else:
                    coords.append([0, 0])
            else:
                coords.append([0, 0])
                
        # [NumPatches, 2] -> [1, NumPatches, 2]
        return torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(self.device)

    def fit(self, train_dataloader, val_dataloader=None, epochs=10, lr=0.001, weight_decay=1e-5, model_save_path=None):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay
            
        scaler = torch.amp.GradScaler('cuda')
        accumulation_steps = 32
        best_auc = 0.0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            self.optimizer.zero_grad()
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
            
            for step, (data, label) in enumerate(progress_bar):
                label = label.to(self.device).float()
                
                if isinstance(data, dict):
                    embeddings = data["embeddings"].to(self.device)
                else:
                    embeddings = data.to(self.device)
                    
                coordinates = self._extract_coordinates(data)
                
                with torch.amp.autocast('cuda'):
                    # Forward pass
                    logits = self.model(embeddings, coordinates)
                    loss = self.loss_fn(logits, label.view(-1, 1)) / accumulation_steps
                
                scaler.scale(loss).backward()
                train_loss += loss.item() * accumulation_steps
                
                if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    
                progress_bar.set_postfix({'loss': f"{train_loss / (step + 1):.4f}"})
                
            if val_dataloader is not None:
                val_auc = self.evaluate(val_dataloader, epoch, epochs)
                if val_auc > best_auc:
                    best_auc = val_auc
                    if model_save_path is not None:
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                        best_path = os.path.join(os.path.dirname(model_save_path), "best_dtfd_model.pth")
                        print(f"--> New best AUC ({best_auc:.4f})! Saving model to {best_path}")
                        self.save_model(best_path)

    def evaluate(self, dataloader, epoch, epochs):
        self.model.eval()
        all_probs = []
        all_labels = []
        val_loss = 0.0
        
        with torch.no_grad():
            desc = f"Epoch {epoch+1}/{epochs} [Val]" if isinstance(epoch, int) else f"Evaluation [{epoch}]"
            progress_bar = tqdm(dataloader, desc=desc)
            for data, label in progress_bar:
                label = label.to(self.device).float()
                
                if isinstance(data, dict):
                    embeddings = data["embeddings"].to(self.device)
                else:
                    embeddings = data.to(self.device)
                    
                coordinates = self._extract_coordinates(data)
                
                with torch.amp.autocast('cuda'):
                    logits = self.model(embeddings, coordinates)
                    loss = self.loss_fn(logits, label.view(-1, 1))
                    val_loss += loss.item()
                    
                prob = torch.sigmoid(logits).item()
                all_probs.append(prob)
                all_labels.append(label.item())
                
                progress_bar.set_postfix({'val_loss': f"{val_loss / len(all_probs):.4f}"})
                
        try:
            fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
        except ValueError:
            optimal_threshold = 0.5
        
        acc, auc, f1, precision, recall = self.save_metrics(all_labels, all_probs, optimal_threshold, epoch)
        print(f"--> Validation Metrics: Acc={acc:.4f}, AUC={auc:.4f}, F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f} (Threshold: {optimal_threshold:.4f})")
        return auc

    def predict(self, test_dataloader):
        print("Starting Prediction on Test Set...")
        self.evaluate(test_dataloader, epoch="test", epochs=1)
        print("Starting Region of Interest (ROI) extraction and visualization...")
        
        visualiser = Visualiser(self.model, self.device, self.metrics_path, self._extract_coordinates)
        visualiser.extract_and_visualise_rois(test_dataloader)

    def save_metrics(self, all_labels, all_probs, threshold, epoch):
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
        
        cm = confusion_matrix(all_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.metrics_path, f'{epoch}_confusion_matrix.png'))
        plt.close()
        
        with open(os.path.join(self.metrics_path, f'{epoch}_metrics.txt'), 'w') as f:
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"Optimal Threshold: {threshold:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")
            
        return acc, auc, f1, precision, recall
        
    def save_model(self, path):
        self.model.save_model(path)

    def load_model(self, path):
        self.model.load_model(path)
