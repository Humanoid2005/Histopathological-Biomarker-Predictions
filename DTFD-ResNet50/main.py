import torch
from dataset import BiomarkerDataset, BiomarkerDataLoader
from pipeline import BioMarkerPredictor

if "__main__" == "__main__":
    TRAIN_DATA_PATH = "/home/pathousr4/sriram-srikanth/resnet50_embeddings"
    # TEST_DATA_PATH = ""
    GT_CSV_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/DTFD-ResNet50/ground_truth.csv"
    BIOMARKER = "p53"
    MODEL_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/DTFD-ResNet50/models/dtfd_model.pth"
    EPOCHS = 10
    LR = 1e-3
    WEIGHT_DECAY = 1e-5
    NUM_BAGS = 5
    SPLITS_DATA_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/DTFD-ResNet50/splits/split_info.csv"
    METRICS_PATH = "/home/pathousr4/sriram-srikanth/Histopathological-Biomarker-Predictions/DTFD-ResNet50/metrics"
    dataset = BiomarkerDataset(GT_CSV_PATH,TRAIN_DATA_PATH,BIOMARKER,SPLITS_DATA_PATH)
    dataset.load_data()

    train_dataloader = BiomarkerDataLoader(dataset, 'train', 1, shuffle=True)
    val_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)
    test_dataloader = BiomarkerDataLoader(dataset, 'val', 1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device",device)

    pipeline = BioMarkerPredictor(device,NUM_BAGS,METRICS_PATH)
    pipeline.fit(train_dataloader, val_dataloader,epochs=EPOCHS,lr=LR,weight_decay=WEIGHT_DECAY)
    # pipeline.save_model(MODEL_PATH)
    pipeline.load_model(MODEL_PATH)
    pipeline.predict(test_dataloader)
