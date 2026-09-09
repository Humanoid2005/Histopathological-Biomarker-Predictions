import os
import shutil
import subprocess
import getpass
from tqdm import tqdm

# ---------------- CONFIGURATION ---------------- #
LOCAL_PATCHES_DIR = r"F:\IPD Brain Dataset\patches"
NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"
NAS_USER = "prof-sushree"

BASE_DIR = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}\sriram-srikanth"
NAS_PATCHES_DIR = os.path.join(BASE_DIR, "patches")
# ----------------------------------------------- #

def connect_to_nas():
    """Authenticates with the NAS using Windows 'net use'."""
    print(f"Authentication required for {NAS_USER}")
    password = getpass.getpass(prompt="Enter password: ")
    
    nas_path = rf"\\{NAS_SERVER_IP}"
    cmd = f'net use "{nas_path}" /user:{NAS_USER} "{password}"'
    
    print(f"Connecting to NAS at {nas_path}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0 or "System error 1219" in result.stderr:
        print("Connected to NAS successfully (or connection already exists).\n")
        return True
    else:
        print(f"Failed to connect. Error: {result.stderr.strip()}")
        return False

def copy_folders_with_progress():
    """Copies the directory tree from local to NAS with a tqdm progress bar."""
    if not os.path.exists(LOCAL_PATCHES_DIR):
        print(f"Error: Source directory '{LOCAL_PATCHES_DIR}' does not exist.")
        return

    print(f"Scanning source directory to count files...")
    # Count total files to set the tqdm max value
    total_files = 1606527 #sum(len(files) for _, _, files in os.walk(LOCAL_PATCHES_DIR))
    
    print(f"Found {total_files} files. Starting copy process...")
    print(f"Source: {LOCAL_PATCHES_DIR}\nDestination: {NAS_PATCHES_DIR}\n")
    
    try:
        # Initialize the tqdm progress bar
        with tqdm(total=total_files, desc="Copying Patches", unit="file") as pbar:
            
            # Define a custom copy function that updates the progress bar
            def copy_and_update(src, dst, **kwargs):
                shutil.copy2(src, dst, **kwargs)
                pbar.update(1)
            
            # Execute copytree using the custom function
            shutil.copytree(
                LOCAL_PATCHES_DIR, 
                NAS_PATCHES_DIR, 
                dirs_exist_ok=True, 
                copy_function=copy_and_update
            )
            
        print("\nCopy operation completed successfully!")
        
    except Exception as e:
        print(f"\nAn error occurred during the copy operation: {e}")

if __name__ == "__main__":
    if connect_to_nas():
        copy_folders_with_progress()