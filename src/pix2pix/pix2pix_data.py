from src.data.dataset import Dataset


class Pix2PixDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        lr = self.load_image(idx, resolution="lr")
        hr = self.load_image(idx, resolution="hr")
        sr = self.interpolate(lr)

        return {"B": hr.float(), "A": sr.float()}
