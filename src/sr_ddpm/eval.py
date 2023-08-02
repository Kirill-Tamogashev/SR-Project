import argparse

from tqdm import tqdm
from ml_collections import ConfigDict
import torch
from torch.utils.data import DataLoader

import submodules.ddpm_sr3.model as Model
import submodules.ddpm_sr3.core.logger as Logger

from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image
from src.data.dataset import Dataset as DataSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='sr3_experiments')
    parser.add_argument('--config', default='my_config/sr_weather.yaml')
    parser.add_argument('--phase', choices=('train', 'val'), default='train')
    parser.add_argument('-g', '--gpu_ids', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    args = parser.parse_args()

    ckpt = torch.load(args.path)
    opt = ConfigDict(Logger.parse(args))
    test_set = DataSet(
        lr_path="val_T2_V10_U10_d02_2019_2020_lr_npy",
        hr_path="val_T2_V10_U10_d02_2019_2020_hr_npy",
        normalize=True
    )

    test_loader = DataLoader(test_set, batch_size=64)
    device = torch.device("cuda:0")
    metrics = Metrics(device)

    diffusion = Model.create_model(opt)
    diffusion.netG.load_state_dict(ckpt)
    diffusion.netG.cuda()
    diffusion.netG.eval()

    diffusion.set_new_noise_schedule(opt.model.beta_schedule.val, schedule_phase="val")

    with torch.no_grad():
        for test_data_batch in tqdm(test_loader, total=len(test_loader)):
            diffusion.feed_data(test_data_batch)

            print(diffusion.conditional)
            diffusion.test(continous=False)
            print(diffusion.SR.shape)

            visuals = diffusion.get_current_visuals()
            sr_img = tensor2image(visuals["SR"], (-1.0, 1.0), device)
            hr_img = tensor2image(visuals["HR"], (-1.0, 1.0), device)

            metrics.update(hr_img, sr_img)

    metrics.print()


if __name__ == '__main__':
    main()