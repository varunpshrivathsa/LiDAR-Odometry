import os 
import numpy as np 

class KITTILoader:
    def __init__(self, sequence_path):
        self.velodyne_path = os.path.join(sequence_path,"velodyne")
        self.files = sorted(os.listdir(self.velodyne_path))

    def __len__(self):
        return len(self.files)
    
    def get_point_cloud(self, idx):
        file_path = os.path.join(self.velodyne_path, self.files[idx])
        points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        return points[:,:3] 