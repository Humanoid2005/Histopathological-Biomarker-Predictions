import cv2
import os
from h5_reader import PatchesReader

# If you are running this on Ubuntu where it's mounted via GIO:
import glob
uid = os.getuid() if hasattr(os, "getuid") else 1000
try:
    server_dir = glob.glob(f"/run/user/{uid}/gvfs/smb-share:server=172.16.202.70*")[0]
    if os.path.exists(os.path.join(server_dir, "home")):
        linux_path = os.path.join(server_dir, "home", "sriram-srikanth", "patches", "IN Brain-0002.h5")
    else:
        linux_path = os.path.join(server_dir, "sriram-srikanth", "patches", "IN Brain-0002.h5")
except IndexError:
    linux_path = "NOT_MOUNTED"

# If you are running this on Windows where you mapped it to Z:
# (You must run: net use Z: \\172.16.202.70\home i!DT7zDG /user:ivanbh /persistent:yes)
windows_path = r"Z:\sriram-srikanth\patches\IN Brain-0002.h5"

# Auto-detect which OS you are running the test on
h5_file = linux_path if os.name == "posix" else windows_path

print(f"Opening {h5_file} directly via OS mount...")

try:
    reader = PatchesReader(h5_path=h5_file, batch_size=4)
    print(f"Success! Total patches mapped in file: {len(reader)}")
    
    if reader.hasNext():
        images, names = reader.next()
        print(f"\nRead 1st batch of {len(images)} patches.")
        print(f"Patch names: {names}")
        print(f"Image 1 shape: {images[0].shape}")
        print(f"Image 1 dtype: {images[0].dtype}")
        
        local_output = "test_extracted_patch.png"
        cv2.imwrite(local_output, images[0])
        print(f"\nSaved the first extracted patch locally as '{local_output}' so you can verify it!")
    
    reader.close()
    print("Test complete.")

except Exception as e:
    print(f"\nError: {e}")
    print("Make sure the NAS is properly mounted to your OS before running this script.")
