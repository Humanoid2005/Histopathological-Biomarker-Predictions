import torch
import numpy as np
import random
from dataset import BiomarkerDataset, BiomarkerDataLoader
from pipeline import BioMarkerPredictor
import os

if __name__ == "__main__":
    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = 42
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    # ── Paths & Hyperparameters ───────────────────────────────────────────────
    TRAIN_DATA_PATH  = "../resnet50_embeddings"   # 2048-dim ResNet50 features
    GT_CSV_PATH      = "../ground_truth.csv"
    BIOMARKER        = "p53"
    MODEL_PATH       = "./models/dtfd_model.pth"
    SPLITS_DATA_PATH = "./splits/split_info.csv"
    METRICS_PATH     = "./metrics"

    EPOCHS       = 40          # High epochs to let the cosine scheduler smooth it out
    LR           = 1e-4        # Very low LR to prevent the attention from collapsing
    WEIGHT_DECAY = 1e-3        # High weight decay (L2 penalty) to stop overfitting
    IN_FEATURES  = 2048        # ResNet50 embedding dimension
    HIDDEN_DIM   = 512         # Larger hidden dim — ResNet50 features are richer than Gigapath
    NUM_BAGS     = 5           # Original DTFD paper bag count
    ACCUM_STEPS  = 8           # Gradient accumulation (effective batch = 8 slides)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = BiomarkerDataset(GT_CSV_PATH, TRAIN_DATA_PATH, BIOMARKER, SPLITS_DATA_PATH)
    dataset.load_data()
    # dataset.analyse() # Optional, prints dataset statistics

    train_dataloader = BiomarkerDataLoader(dataset, 'train', 1, shuffle=True)
    val_dataloader   = BiomarkerDataLoader(dataset, 'val',   1, shuffle=False)
    test_dataloader  = BiomarkerDataLoader(dataset, 'val',   1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    # ── Train ─────────────────────────────────────────────────────────────────
    pipeline = BioMarkerPredictor(
        device, NUM_BAGS, METRICS_PATH,
        in_features=IN_FEATURES,
        hidden_dim=HIDDEN_DIM
    )
    pipeline.fit(
        train_dataloader, val_dataloader,
        epochs=EPOCHS, lr=LR,
        weight_decay=WEIGHT_DECAY,
        accum_steps=ACCUM_STEPS,
        model_save_path=MODEL_PATH
    )
    pipeline.save_model(MODEL_PATH)

    # ── Inference with best checkpoint ────────────────────────────────────────
    best_model_path = os.path.join(os.path.dirname(MODEL_PATH), "best_dtfd_model.pth")
    pipeline.load_model(best_model_path if os.path.exists(best_model_path) else MODEL_PATH)
    pipeline.predict(test_dataloader)
