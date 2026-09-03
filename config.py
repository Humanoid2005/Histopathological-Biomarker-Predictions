import os

NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"

# Base UNC path for Windows
BASE_UNC = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}\sriram-srikanth"

RAW_WSI_SLIDES_PATH = os.path.join(BASE_UNC, "images")
PATCHES_PATH = os.path.join(BASE_UNC, "patches")

# Run net use \\172.16.201.2 /user:prof-sushree * on Windows 
# Run gio mount "smb://172.16.201.2/prof-sushree" on Linux