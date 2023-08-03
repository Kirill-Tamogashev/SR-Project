from src.data.dataset import Dataset


class Pix2PixDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        lr = self.load_image(idx, resolution="lr")
        hr = self.load_image(idx, resolution="hr")
        sr = self.interpolate(lr)

        hr = hr.div(127.5).sub(1)
        sr = sr.div(127.5).sub(1)
            
        return {"B": hr.float(), "A": sr.float(), "A_paths": "", "B_paths": ""}
