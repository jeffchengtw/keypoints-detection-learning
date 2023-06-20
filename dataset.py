import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data = []  
        self.transform = transform
        
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith(".bmp") or file.endswith(".jpg") or file.endswith(".png"):
                    file_path = os.path.join(root, file)
                    self.data.append(file_path)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        file_path = self.data[index]
        
        image = np.array(cv2.imread(file_path, cv2.IMREAD_UNCHANGED))
        resized_image = cv2.resize(image, (512, 512))
        
        file_name = os.path.basename(file_path)
        filename, extention = os.path.splitext(file_name)
        bgr_img = resized_image.copy()
        gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        
        if self.transform is not None:
            image = self.transform(image)
        
        gray_tensor = torch.from_numpy(gray_img).float()
        # Add a new dimension
        gray_tensor = gray_tensor.unsqueeze(0)
        return {
            'filename' : filename,
            'gray_tensor' : gray_tensor,
            'bgr_arr' : bgr_img
        }
