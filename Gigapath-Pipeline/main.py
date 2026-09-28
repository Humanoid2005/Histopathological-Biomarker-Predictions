import torch
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import BiomarkerDataset, BiomarkerDataLoader
from pipeline import BioMarkerPredictor

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

    # Using the flash embeddings paths
    TRAIN_DATA_PATH = "/home/pathousr4/sriram-srikanth/gigapath-flash-embeddings"
    GT_CSV_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/ground_truth.csv"
    BIOMARKER = "p53"
    MODEL_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/Gigapath-Pipeline/models/flash_model.pth"
    SLIDE_ENCODER_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/Gigapath-Pipeline/models/slide_encoder.pth"
    EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    SPLITS_DATA_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/Gigapath-Pipeline/splits/split_info.csv"
    METRICS_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/Gigapath-Pipeline/metrics_flash"
    
    dataset = BiomarkerDataset(GT_CSV_PATH, TRAIN_DATA_PATH, BIOMARKER, SPLITS_DATA_PATH)
    dataset.load_data()

    train_dataloader = BiomarkerDataLoader(dataset, 'train', 1, shuffle=True)
    val_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)
    test_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # Note: BioMarkerPredictor now takes slide_encoder_path
    pipeline = BioMarkerPredictor(device, METRICS_PATH, slide_encoder_path=SLIDE_ENCODER_PATH)
    pipeline.fit(train_dataloader, val_dataloader, epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, model_save_path=MODEL_PATH)
    pipeline.save_model(MODEL_PATH)
    
    # Load the best model if it was saved during training
    best_model_path = os.path.join(os.path.dirname(MODEL_PATH), "best_dtfd_model.pth")
    if os.path.exists(best_model_path):
        pipeline.load_model(best_model_path)
    else:
        pipeline.load_model(MODEL_PATH)
        
    pipeline.predict(test_dataloader)
