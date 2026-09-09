import os
import shutil
import subprocess

# --- Configuration Setup ---
NAS_SERVER_IP = "172.16.201.2"
SHARE_NAME = "prof-sushree"
BASE_DIR = rf"\\{NAS_SERVER_IP}\{SHARE_NAME}"

# Using SHARE_NAME as username as per your config
NAS_USER = SHARE_NAME
NAS_PASS = "5Qmm*P"

def authenticate_nas():
    """Authenticates to the NAS using built-in Windows commands."""
    command = f'net use "{BASE_DIR}" /user:{NAS_USER} {NAS_PASS}'
    # We suppress output here to keep your console clean
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Ignore errors if we are already connected to it
    if result.returncode != 0 and "multiple connections" not in result.stderr.lower():
        print(f"Notice during NAS authentication: {result.stderr.strip()}")

def get_folder_size_basic(root_share_path, target_subfolder):
    """Recursively crawls a standard OS directory and sums up file sizes."""
    total_size = 0
    clean_subfolder = target_subfolder.replace("/", "\\").strip("\\")
    full_folder_path = os.path.join(root_share_path, clean_subfolder)
    
    try:
        for dirpath, dirnames, filenames in os.walk(full_folder_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                try:
                    # os.path.getsize is much faster natively
                    if not os.path.islink(file_path):
                        total_size += os.path.getsize(file_path)
                except Exception:
                    continue  # Skip files that are locked or inaccessible
    except Exception as e:
        print(f"Warning: Could not read folder '{target_subfolder}': {e}")
        
    return total_size

def get_nas_and_folder_details(target_folders):
    try:
        # 1. Authenticate natively via OS
        authenticate_nas()
        
        # 2. Get Overall Volume Storage using standard Python shutil
        total_bytes, used_bytes, free_bytes = shutil.disk_usage(BASE_DIR)
        
        bytes_to_gb = 1024 ** 3
        bytes_to_mb = 1024 ** 2
        
        print(f"=== OVERALL NAS VOLUME CAPACITY ({SHARE_NAME}) ===")
        print(f"Total Capacity : {total_bytes / bytes_to_gb:.2f} GB")
        print(f"Free Space     : {free_bytes / bytes_to_gb:.2f} GB")
        print(f"Total Used     : {used_bytes / bytes_to_gb:.2f} GB")
        print(f"Usage          : {(used_bytes / total_bytes) * 100:.2f}%\n")

        # 3. Check Specific Target Directory Footprints
        print(f"=== TARGET DIRECTORY FOOTPRINTS ===")
        for folder in target_folders:
            size_in_bytes = get_folder_size_basic(BASE_DIR, folder)
            
            if size_in_bytes >= bytes_to_gb:
                print(f"Folder '{folder}' Size: {size_in_bytes / bytes_to_gb:.2f} GB")
            else:
                print(f"Folder '{folder}' Size: {size_in_bytes / bytes_to_mb:.2f} MB")
                
    except Exception as e:
        print(f"Failed to pull NAS metrics: {e}")

# Paths relative to your share root
folders_to_check = [
    "sriram-srikanth",
    r"sriram-srikanth\patches"
]

# Run the execution pipeline
if __name__ == "__main__":
    get_nas_and_folder_details(folders_to_check)