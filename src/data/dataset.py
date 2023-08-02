from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F

BASE_PATH = Path("/mnt/NFS/labutin_datasets")


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self,
            lr_path: str,
            hr_path: str,
            low_res: int = 64,
            high_res: int = 128,
    ):

        self.lr_path = BASE_PATH / lr_path
        self.hr_path = BASE_PATH / hr_path
        self.low_res = low_res
        self._high_res = high_res

        self.lr_images = self.from_dataset(resolution="lr")
        self.hr_images = self.from_dataset(resolution="hr")

        assert len(self.hr_images) == len(self.lr_images), \
            "Low res and High res data folders must have the same " \
            "number of images"

    def from_dataset(self, resolution: str):
        path = self.hr_path if resolution == "hr" else self.lr_path
        return sorted([image_path.as_posix() for image_path in path.glob("*.npy")])

    @staticmethod
    def interpolate(x):
        return F.interpolate(
            x.unsqueeze(0).float(), size=(128, 128),
            mode='bilinear', align_corners=True
        ).squeeze()

    def load_image(self, idx: int, resolution: str) -> torch.Tensor:
        image = np.load(self.hr_images[idx]) if resolution == "hr" else np.load(self.lr_images[idx])
        return torch.tensor(image).permute(2, 0, 1)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        lr = self.load_image(idx, resolution="lr")
        hr = self.load_image(idx, resolution="hr")
        sr = self.interpolate(lr)

        return {"HR": hr.float(), "SR": sr.float(), "LR": lr.float(), "Index": idx}
