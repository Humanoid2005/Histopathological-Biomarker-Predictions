import os

NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"

# Run net use \\172.16.201.2 /user:prof-sushree * on Windows
'''
BASE_UNC = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}\sriram-srikanth"
RAW_WSI_SLIDES_PATH = os.path.join(BASE_UNC, "images")
PATCHES_PATH = os.path.join(BASE_UNC, "patches")
'''

# Run gio mount "smb://172.16.201.2/prof-sushree" on Linux
uid = os.getuid()
GVFS_ROOT = f"/run/user/{uid}/gvfs/smb-share:server={NAS_SERVER_IP},share={SHARE_NAME}"
BASE_DIR = os.path.join(GVFS_ROOT, "sriram-srikanth")
RAW_WSI_SLIDES_PATH = os.path.join(BASE_DIR, "images")
PATCHES_PATH = os.path.join(BASE_DIR, "patches")

