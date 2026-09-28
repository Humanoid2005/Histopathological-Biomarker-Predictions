import os

NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"

BASE_DIR = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}\sriram-srikanth"
# uid = os.getuid()
# GVFS_ROOT = f"/run/user/{uid}/gvfs/smb-share:server={NAS_SERVER_IP},share={SHARE_NAME}"
# BASE_DIR = os.path.join(GVFS_ROOT, "sriram-srikanth")
PATCHES_PATH = r"F:\IPD Brain Dataset\patches"
EMBEDDINGS_PATH = "../gigapath-flash-embeddings"
MODEL_PATH = "./models"