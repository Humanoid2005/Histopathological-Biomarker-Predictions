import os

NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"

# Run net use \\172.16.201.2 /user:prof-sushree * on Windows
# Run gio mount "smb://172.16.201.2/prof-sushree" on Linux

BASE_DIR = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}\sriram-srikanth"
# uid = os.getuid()
# GVFS_ROOT = f"/run/user/{uid}/gvfs/smb-share:server={NAS_SERVER_IP},share={SHARE_NAME}"
# BASE_DIR = os.path.join(GVFS_ROOT, "sriram-srikanth")
RAW_WSI_SLIDES_PATH = os.path.join(BASE_DIR, "images")
PATCHES_PATH = os.path.join(BASE_DIR, "patches")
# EMBEDDINGS_PATH = os.path.join(BASE_DIR, "embeddings")
EMBEDDINGS_PATH = "../resnet50_embeddings"  # Path updated according to the user's ls output
#RAW_WSI_SLIDES_PATH = r"E:\IPD Brain Dataset\images"
#PATCHES_PATH = r"F:\IPD Brain Dataset\patches"

# DTFD-MIL Configuration
NUM_CLASSES = 1 # Binary classification for ATRX
IN_FEATURES = 2048 # Restored this because train.py needs it!
HIDDEN_DIM = 512
PATCHES_PER_PSEUDO_BAG = 1000 # Determines number of pseudo-bags dynamically (num_bags = num_patches // 100)

# Training Config
K_FOLDS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50