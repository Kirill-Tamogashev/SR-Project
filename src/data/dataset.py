from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
import torchvision 
from PIL import Image


BASE_PATH = Path("/mnt/NFS/labutin_datasets")


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            lr_path:        Path, 
            hr_path:        Path, 
            low_res:        int = 64,
            high_res:       int = 128,
            data_format:    str = "npy", 
        ):
        assert data_format == "npy"
        
        self._lr_path = BASE_PATH / lr_path
        self._hr_path = BASE_PATH / hr_path
        self._low_res =  low_res
        self._high_res = high_res
        self._format = data_format
        
        self._totensor = torchvision.transforms.ToTensor()
        self._lr_images = self._from_dataset(resolution="lr")
        self._hr_images = self._from_dataset(resolution="hr")
        
        assert len(self._hr_images) == len(self._lr_images), \
            "Low res and High res data folders must have the same "\
            "number of images"
            
            
    def _from_dataset(self, resolution: str):
        path = self._hr_path if resolution == "hr" else self._lr_path
        return sorted([
            image_path for image_path 
            in path.glob(f"./*.{self._format}")
        ])

    def __len__(self):
        return len(self._hr_images)
    
    def __getitem__(self, idx):
        lr_image = np.load(self._lr_images[idx])
        hr_image = np.load(self._hr_images[idx])
            
        LR = torch.tensor(lr_image).permute(2, 0, 1)
        HR = torch.tensor(hr_image).permute(2, 0, 1)
        SR = F.interpolate(
            LR.unsqueeze(0).float(), size=(128, 128), 
            mode='bilinear', align_corners=True
        ).squeeze()
        return {"HR": HR.float(), "SR": SR.float(), "LR": LR.float(), "Index": idx}