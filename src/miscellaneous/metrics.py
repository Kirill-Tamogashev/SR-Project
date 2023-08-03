from typing import Optional

import torch

from tqdm import tqdm

from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.inception import InceptionScore as IS
from torchmetrics import PeakSignalNoiseRatio as PSNR
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics import MeanSquaredError as L2
from torchmetrics import MeanAbsoluteError as L1
from prettytable import PrettyTable


class Metrics:
    def __init__(self, device):
        self.metric_dict = {
            "L1": L1().to(device),
            "L2": L2().to(device),
            "Channel 1": L2().to(device),
            "Channel 2": L2().to(device),
            "Channel 3": L2().to(device),
            "FID": FID().to(device),
            "LPIPS": LPIPS().to(device),
            "IS": IS().to(device),
            "PSNR": PSNR().to(device),
            "SSIM": SSIM().to(device),
        }
        if device != "cpu":
            self.to(device)

    def to(self, device):
        for name in self.metric_dict.keys():
            self.metric_dict[name] = self.metric_dict[name].to(device)

    @staticmethod
    def get_channel_slice(name, real, fake):
        if name.startswith("Channel"):
            idx = int(name.split()[-1]) - 1
            fake, real = fake[:, idx, :, :], real[:, idx, :, :]
        return real, fake

    def update(self, real: torch.Tensor, fake: torch.Tensor):
        for name in self.metric_dict.keys():
            if name == "IS":
                self.metric_dict[name].update(fake)
            elif name == "FID":
                self.metric_dict[name].update(real, real=True)
                self.metric_dict[name].update(fake, real=False)
            elif name in {"LPIPS", "SSIM"}:
                sr_normed = torch.clamp(fake / 255 * 2 - 1, -1, 1)
                hr_normed = torch.clamp(real / 255 * 2 - 1, -1, 1)
                self.metric_dict[name].update(sr_normed, hr_normed)
            else:
                real_sliced, fake_sliced = self.get_channel_slice(name, real, fake)
                self.metric_dict[name].update(fake_sliced, real_sliced)
                
    def compute(self, rounding: Optional[int] = None):
        value_dict = {}
        for name, metric in tqdm(self.metric_dict.items(), desc="Computing metrics", leave=False):
            value = metric.compute()[0] if name == "IS" else metric.compute()
            value_dict[name] = value.item() if rounding is None else round(value.item(), rounding)
        return value_dict

    def print(self):
        tab = PrettyTable(["Metric name", "Metric value"])
        value_dict = self.compute(rounding=3)
        for name, value, in value_dict.items():
            tab.add_row([name, value])
        print(tab)
        
    def reset(self):
        for name in self.metric_dict.keys():
            self.metric_dict[name].reset()
