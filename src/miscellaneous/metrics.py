import torch

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

    def print(self):
        tab = PrettyTable(["Metric name", "Metric value"])
        for name, metric in self.metric_dict.items():
            if name == "IS":
                value = metric.compute()[0]
            else:
                value = metric.compute()

            value = value.item()
            tab.add_row([name, round(value, 3)])

        print(tab)
