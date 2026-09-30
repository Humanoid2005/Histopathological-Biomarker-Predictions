import smbclient
import cv2
import tempfile
import os
from tqdm import tqdm
from h5_reader import PatchesReader

# --- Configuration ---
OUT_NAS_IP = "172.16.202.70"
OUT_SHARE = "home"
OUT_USER = "ivanbh"
OUT_PASS = "i!DT7zDG"

# Crucial: Prevent the NAS from forcefully dropping the connection during long 5GB downloads
smbclient.ClientConfig(session_timeout=36000)

print("Authenticating with NAS via smbclient...")
smbclient.register_session(OUT_NAS_IP, username=OUT_USER, password=OUT_PASS)

remote_h5_file = rf"\\{OUT_NAS_IP}\{OUT_SHARE}\sriram-srikanth\patches\IN Brain-0002.h5"
print(f"Connecting to NAS and opening {remote_h5_file}...")

try:
    with tempfile.TemporaryDirectory() as temp_dir:
        local_h5 = os.path.join(temp_dir, "IN Brain-0002.h5")
        
        print(f"Downloading {remote_h5_file} locally for safe h5py access...")
        file_size = smbclient.stat(remote_h5_file).st_size
        
        with smbclient.open_file(remote_h5_file, mode="rb") as f_src:
            with open(local_h5, "wb") as f_dst:
                with tqdm(total=file_size, unit="B", unit_scale=True, desc="Downloading H5") as pbar:
                    while chunk := f_src.read(1024 * 1024):
                        f_dst.write(chunk)
                        pbar.update(len(chunk))
        
        print("File downloaded successfully! Initializing PatchesReader...")
        
        reader = PatchesReader(h5_path=local_h5, batch_size=4)
        print(f"Success! Total patches mapped in file: {len(reader)}")
        
        if reader.hasNext():
            images, names = reader.next()
            print(f"\nRead 1st batch of {len(images)} patches.")
            print(f"Patch names: {names}")
            print(f"Image 1 shape: {images[0].shape}")
            
            local_output = "test_extracted_patch.png"
            cv2.imwrite(local_output, images[0])
            print(f"\nSaved the first extracted patch locally as '{local_output}' so you can verify it!")
        
        reader.close()
        print("Test complete.")

except Exception as e:
    print(f"\nError: {e}")
    print("If you get a File Not Found error, IN Brain-0002.h5 hasn't finished uploading yet.")
