from argparse import ArgumentParser
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import gc

import torch.nn.functional as F

from submodules.torch_not.src.resnet2 import ResNet_D
from submodules.torch_not.src.unet import UNet
from submodules.torch_not.src.tools import unfreeze, freeze
from submodules.torch_not.src.tools import weights_init_D
from submodules.torch_not.src.plotters import plot_images
# from submodules.torch_not.src.custom_data import load_paired_data_for_train, load_paired_data_for_test
from submodules.torch_not.src.tools import fig2img

from tqdm.notebook import tqdm
from IPython.display import clear_output

import wandb
from src.miscellaneous.losses import VGGPerceptualLoss as VGGLoss
from src.neural_ot.unet2 import U2NET
from src.constants import BaseCheckpoint


# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin

SEED = 0x000000
DATA_SR = "sr"
DATA_HR = "hr"


def parse_arguments():
    parser = ArgumentParser(description='NOT for SR')
    parser.add_argument('--max_steps', type=int, default=10_001)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--t_iters', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--f_lr', type=float, default=1e-4)
    parser.add_argument('--t_lr', type=float, default=1e-4)
    parser.add_argument('--gpu_ids', type=str, default="0")
    parser.add_argument('--cpkt_interval', type=int, default=500)
    parser.add_argument('--plot_interval', type=int, default=100)
    parser.add_argument('--cost', type=str, default="mse", choices=("mse", "vgg"))
    parser.add_argument('--model', type=str, default="unet", choices=("unet", "unet2"))
    parser.add_argument('--wandb', action="store_true", help="Use wandb")
    parser.add_argument('--superwised', action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

    args = parse_arguments()
    DEVICE_IDS = [int(gpu_id) for gpu_id in args.gpu_ids.split(",")]

    EXP_NAME = f"not-superwised-{args.model}-{args.cost}"
    print(f"Experiment name is {EXP_NAME}")
    OUTPUT_PATH = f'../checkpoints/{EXP_NAME}'

    config = dict(
        DATASET1=DATA_SR,
        DATASET2=DATA_HR,
        T_ITERS=args.t_iters,
        f_LR=args.f_lr,
        T_LR=args.t_lr,
        BATCH_SIZE=args.batch_size
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(f'cuda:{DEVICE_IDS[0]}')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    elif len(os.listdir(OUTPUT_PATH)) != 0:
        raise AssertionError(f"{OUTPUT_PATH} is not an empty directory. Training is impossible")

    paired_sampler = load_paired_data_for_train(BATCH_SIZE)
    paired_test_loader = load_paired_data_for_test(10)
    torch.cuda.empty_cache();
    gc.collect()

    f = ResNet_D(args.img_size, nc=3).cuda()
    f.apply(weights_init_D)

    T = U2NET(in_ch=3, out_ch=3).cuda() if args.model == "unet2" else UNet(3, 3, base_factor=48).cuda()

    if len(DEVICE_IDS) > 1:
        T = nn.DataParallel(T, device_ids=DEVICE_IDS)
        f = nn.DataParallel(f, device_ids=DEVICE_IDS)

    torch.manual_seed(0xBADBEEF);
    np.random.seed(0xBADBEEF)

    X_fixed, Y_fixed = paired_sampler.sample(10)
    test_batch = next(iter(paired_test_loader))
    X_test_fixed, Y_test_fixed = test_batch["SR"], test_batch["HR"]

    T_opt = torch.optim.Adam(T.parameters(), lr=args.t_lr, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=args.f_lr, weight_decay=1e-10)

    loss_fn = VGGLoss().cuda() if args.cost == "vgg" else F.mse_loss

    with wandb.init(name=EXP_NAME, project='SuperResWithNOT', config=config):
        print("Start running the train loop")
        for step in tqdm(range(args.max_step)):
            # T optimization
            unfreeze(T)
            freeze(f)
            for t_iter in range(args.t_iters):
                T_opt.zero_grad()
                # X = X_sampler.sample(BATCH_SIZE)
                X, Y = paired_sampler.sample(args.batch_size)
                T_X = T(X)
                T_loss = loss_fn(Y, T_X).mean() - f(T_X).mean()
                T_loss.backward()
                T_opt.step()
            del T_loss, T_X, X
            gc.collect()
            torch.cuda.empty_cache()

            # f optimization
            freeze(T)
            unfreeze(f)

            if args.superwised:
                X, Y = paired_sampler.sample(args.batch_size)
            else:
                X = X_sampler.sample(BATCH_SIZE)
                Y = Y_sampler.sample(BATCH_SIZE)

            with torch.no_grad():
                T_X = T(X)
            f_opt.zero_grad()
            f_loss = f(T_X).mean() - f(Y).mean()
            f_loss.backward()
            f_opt.step()
            wandb.log({f'f_loss': f_loss.item()}, step=step)
            del f_loss, Y, X, T_X
            gc.collect()
            torch.cuda.empty_cache()

            if step % args.plot_interval == 0:
                print('Plotting')
                clear_output(wait=True)

                fig, axes = plot_images(X_fixed, Y_fixed, T)
                wandb.log({'Fixed Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

                X_random, Y_random = paired_sampler.sample(10)
                fig, axes = plot_images(X_random, Y_random, T)
                wandb.log({'Random Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

                # print(X_test_fixed.shape, Y_test_fixed.shape)
                fig, axes = plot_images(X_test_fixed.cuda(), Y_test_fixed.cuda(), T)
                wandb.log({'Fixed Test Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

            if step % args.cpkt_interval == args.cpkt_interval - 1:
                freeze(T)
                torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))

            gc.collect();
            torch.cuda.empty_cache()
