from torch.utils.data import DataLoader

from submodules.torch_not.src.distributions import Sampler
from src.data.dataset import Dataset


class DataSampler(Sampler):
    def __init__(self, loader, device='cuda'):
        super().__init__(device)

        self.loader = loader
        self.it = iter(self.loader)

    def sample(self, size=5):
        assert size <= self.loader.batch_size

        try:
            batch = next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return self.sample(size)
        if len(batch) < size:
            return self.sample(size)

        return batch

    def sample_x(self, size=5):
        batch = self.sample(size)
        return batch["SR"][:size].to(self.device)

    def sample_y(self, size=5):
        batch = self.sample(size)
        return batch["HR"][:size].to(self.device)

    def sample_paired(self, size=5):
        batch = self.sample(size)
        sr = batch["SR"][:size].to(self.device)
        hr = batch["HR"][:size].to(self.device)
        return sr, hr


def load_train_sampler(batch_size: int = 100, device="cuda"):
    lr_path = "train_T2_V10_U10_d02_2017-2019_lr_npy"
    hr_path = "train_T2_V10_U10_d02_2017-2019_hr_npy"
    return DataSampler(
        DataLoader(
            Dataset(lr_path=lr_path, hr_path=hr_path),
            batch_size=batch_size,
            shuffle=True
        ),
        device=device
    )


def load_val_sampler(batch_size: int = 100, device="cuda"):
    lr_path = "val_T2_V10_U10_d02_2019_2020_lr_npy"
    hr_path = "val_T2_V10_U10_d02_2019_2020_hr_npy"
    return DataSampler(
        DataLoader(
            Dataset(lr_path=lr_path, hr_path=hr_path),
            batch_size=batch_size,
            shuffle=True
        ),
        device=device
    )


def load_test_sampler(batch_size: int = 100, device="cuda"):
    lr_path = "test_T2_V10_U10_d02_2021_lr_npy"
    hr_path = "test_T2_V10_U10_d02_2021_hr_npy"
    return DataSampler(
        DataLoader(
            Dataset(lr_path=lr_path, hr_path=hr_path),
            batch_size=batch_size,
            shuffle=True
        ),
        device=device
    )
