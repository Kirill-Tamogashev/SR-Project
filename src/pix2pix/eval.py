from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import sys
sys.path.append("submodules/gan_pix2pix")

from submodules.gan_pix2pix.options.test_options import TestOptions
from submodules.gan_pix2pix.models import create_model

from src.constants import BaseCheckpoint
from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image, load_ckpt
from src.pix2pix.pix2pix_data import Pix2PixDataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main():
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 128  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    opt.eval = True
    
    # Set correct params
    opt.model = "pix2pix"
    opt.netG = "unet_128"
    opt.checkpoints_dir = BaseCheckpoint.PIX2PIX
    
    print(opt)
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
    

    data = Pix2PixDataset(
        lr_path="train_T2_V10_U10_d02_2017-2019_lr_npy",
        hr_path="train_T2_V10_U10_d02_2017-2019_hr_npy",
    )
    dataset = DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads)
    )
    
    model = create_model(opt)
    model_state = load_ckpt(BaseCheckpoint.PIX2PIX / opt.name, device, pattern="*.pth")
    model.netG.module.load_state_dict(model_state)
    model.netG.to(device)
    model.eval()
    
    metrics = Metrics(device)

    if opt.eval:
        model.eval()

    for data in tqdm(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference

        real = tensor2image(model.real_B, (-1.0, 1.0), device)
        fake = tensor2image(model.fake_B, (-1.0, 1.0), device)
        metrics.update(real=real, fake=fake)
    metrics.print()


if __name__ == '__main__':
    main()
