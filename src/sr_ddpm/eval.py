import argparse
import os
from pathlib import Path

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from submodules.ddpm_sr3.model import create_model

from src.constants import BaseCheckpoint
from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image, load_ckpt
from src.data.dataset import Dataset as DataSet
from src.sr_ddpm.utils import configure_params

# Fix local imports within the submodule
import sys
import warnings

sys.path.append("submodules/ddpm_sr3")
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, choices=os.listdir(BaseCheckpoint.SR3))
    parser.add_argument("--config", type=Path, default="./src/sr_ddpm/sr_config.yaml")
    parser.add_argument("--device", type=int, default=None)

    # The following args should not be changed, they are kept for compatability
    parser.add_argument("--phase", default="eval")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--enable_wandb", action="store_true")
    parser.add_argument("--log_wandb_ckpt", action="store_true")
    parser.add_argument("--log_eval", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    opt = configure_params(args)
    device = torch.device(f"cuda:{args.device}")

    model_state = load_ckpt(opt.path.experiments_root, device, pattern="*.pth")

    test_set = DataSet(
        lr_path="val_T2_V10_U10_d02_2019_2020_lr_npy",
        hr_path="val_T2_V10_U10_d02_2019_2020_hr_npy",
        normalize=True
    )

    test_loader = DataLoader(test_set, batch_size=64)
    metrics = Metrics(device)

    diffusion = create_model(opt)
    diffusion.netG.load_state_dict(model_state)
    diffusion.netG.cuda()
    diffusion.netG.eval()

    diffusion.set_new_noise_schedule(opt.model.beta_schedule.val, schedule_phase="val")

    with torch.no_grad():
        for test_data_batch in tqdm(test_loader, total=len(test_loader)):
            diffusion.feed_data(test_data_batch)

            diffusion.test(continous=False)

            visuals = diffusion.get_current_visuals()
            sr_img = tensor2image(visuals["SR"], (-1.0, 1.0), device)
            hr_img = tensor2image(visuals["HR"], (-1.0, 1.0), device)

            metrics.update(hr_img, sr_img)

    metrics.print()


if __name__ == '__main__':
    main()
