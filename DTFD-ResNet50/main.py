import torch
from dataset import BiomarkerDataset, BiomarkerDataLoader
from pipeline import BioMarkerPredictor

if "__main__" == "__main__":
    torch.manual_seed(42)
    import random
    random.seed(42)

    # TRAIN_DATA_PATH = "../resnet50_embeddings"
    TRAIN_DATA_PATH = "/home/pathousr4/sriram-srikanth/gigapath-flash-embeddings"
    # TEST_DATA_PATH = ""
    GT_CSV_PATH = "../ground_truth.csv"
    BIOMARKER = "p53"
    MODEL_PATH = "./models/dtfd_model.pth"
    EPOCHS = 10
    LR = 1e-3
    IN_FEATURES = 384
    HIDDEN_DIM = 256
    WEIGHT_DECAY = 1e-5
    NUM_BAGS = 5
    SPLITS_DATA_PATH = "./splits/split_info.csv"
    METRICS_PATH = "./metrics"
    dataset = BiomarkerDataset(GT_CSV_PATH,TRAIN_DATA_PATH,BIOMARKER,SPLITS_DATA_PATH)
    dataset.load_data()

    train_dataloader = BiomarkerDataLoader(dataset, 'train', 1, shuffle=True)
    val_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)
    test_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device",device)

    pipeline = BioMarkerPredictor(device,NUM_BAGS,METRICS_PATH,IN_FEATURES,HIDDEN_DIM)
    pipeline.fit(train_dataloader, val_dataloader,epochs=EPOCHS,lr=LR,weight_decay=WEIGHT_DECAY, model_save_path=MODEL_PATH)
    pipeline.save_model(MODEL_PATH)
    
    # Load the best model if it was saved during training
    import os
    best_model_path = os.path.join(os.path.dirname(MODEL_PATH), "best_dtfd_model.pth")
    if os.path.exists(best_model_path):
        pipeline.load_model(best_model_path)
    else:
        pipeline.load_model(MODEL_PATH)
        
    pipeline.predict(test_dataloader)
