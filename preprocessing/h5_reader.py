import h5py
import numpy as np
import cv2

class PatchesReader:
    def __init__(self, h5_path=None, fileobj=None, batch_size=1):
        self.h5_path = h5_path
        self.batch_size = batch_size
        if fileobj is not None:
            self.file = h5py.File(fileobj, 'r')
        else:
            self.file = h5py.File(h5_path, 'r')
        
        # Load metadata arrays into RAM for ultra-fast indexing
        self.offsets = self.file['offsets'][:]
        self.lengths = self.file['lengths'][:]
        self.filenames = self.file['filenames'][:]
        self.image_bytes = self.file['image_bytes']
        
        self.num_patches = len(self.offsets)
        self.current_idx = 0
        
    def __len__(self):
        return self.num_patches

    def hasNext(self):
        return self.current_idx < self.num_patches

    def next(self):
        if not self.hasNext():
            raise StopIteration("No more patches available in the H5 file.")
            
        end_idx = min(self.current_idx + self.batch_size, self.num_patches)
        
        batch_images = []
        batch_names = []
        
        for i in range(self.current_idx, end_idx):
            offset = self.offsets[i]
            length = self.lengths[i]
            
            # Extract raw bytes from the single HDF5 array
            raw_bytes = self.image_bytes[offset : offset + length]
            
            # Decode the raw PNG bytes back into a NumPy array image
            img_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            batch_images.append(img)
            batch_names.append(self.filenames[i].decode('utf-8'))
            
        self.current_idx = end_idx
        
        # If batch_size is 1, just return the single image instead of a list
        if self.batch_size == 1:
            return batch_images[0], batch_names[0]
            
        return batch_images, batch_names
        
    def __iter__(self):
        self.current_idx = 0
        return self
        
    def __next__(self):
        if not self.hasNext():
            raise StopIteration
        return self.next()

    def close(self):
        self.file.close()


if __name__ == "__main__":
    pass
