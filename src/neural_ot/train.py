from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import gc

import torch.nn.functional as F

from submodules.torch_not.src.resnet2 import ResNet_D
from submodules.torch_not.src.unet import UNet
from submodules.torch_not.src.tools import unfreeze, freeze
from submodules.torch_not.src.tools import weights_init_D
from submodules.torch_not.src.plotters import plot_images
from submodules.torch_not.src.tools import fig2img

from tqdm import trange
from IPython.display import clear_output

import wandb
from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image
from src.miscellaneous.losses import VGGPerceptualLoss as VGGLoss
from src.neural_ot.unet2 import U2NET
from src.neural_ot.data_samplers import load_train_sampler, load_val_sampler
from src.constants import BaseCheckpoint

# This needed to use dataloaders for some datasets
from PIL import PngImagePlugin

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


SEED = 0x000000
DATA_SR = "sr"
DATA_HR = "hr"


def parse_arguments():
    parser = ArgumentParser(
        description='NOT for SR', formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name",           help="Additional name of the run", type=str, default=None)
    
    parser.add_argument("--max_steps",      help="Number of max steps", type=int, default=10_001)
    parser.add_argument("--batch_size",     help="Batch size for traning", type=int, default=64)
    parser.add_argument("--test_batch",     help="Batch size for testing", type=int, default=50)
    parser.add_argument("--test_ites",      help="Numbet of testing iterations", type=int, default=100)
    parser.add_argument("--t_iters",        help="Number of iterations for T", type=int, default=10)
    parser.add_argument("--img_size",       help="Size of image", type=int, default=128)
    parser.add_argument("--f_lr",           help="Learning rate for f", type=float, default=1e-4)
    parser.add_argument("--t_lr",           help="Learning rate for T", type=float, default=1e-4)
    parser.add_argument("--gpu_id",         help="Device for traning", type=str, default=None)
    
    parser.add_argument("--cpkt_interval",  help="Frequency of checkpointing", type=int, default=2000)
    parser.add_argument("--plot_interval",  help="Frequency of plotting", type=int, default=1000)
    parser.add_argument("--metric_freq",    help="Frequency of metric printing", type=int, default=200)
    
    parser.add_argument("--cost",           help="Cost function C", type=str, default="mse", choices=("mse", "vgg"))
    parser.add_argument("--model",          help="Model type", type=str, default="unet", choices=("unet", "unet"))
    
    parser.add_argument("--dry-run",     action="store_true", help="Disable wandb")
    parser.add_argument("--supervised",     action="store_true", help="Enable superwised OT")
    return parser.parse_args()


def main():
    LARGE_ENOUGH_NUMBER = 100
    PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024 ** 2)

    args = parse_arguments()
    device = torch.device(f"cuda:{args.gpu_id}" if args.gpu_id is not None else "cpu")

    NOT_TYPE = "supervised" if args.supervised else "unsupervised"
    EXP_NAME = f"not-{NOT_TYPE}-{args.model}-{args.cost}"
    if args.name is not None:
        EXP_NAME = f"{EXP_NAME}-{args.name}"
    OUTPUT_PATH = BaseCheckpoint.NOT / EXP_NAME

    print(f"Experiment name is {EXP_NAME}")
    print(f"Output Path is {OUTPUT_PATH.as_posix()}")

    config = dict(
        DATASET1=DATA_SR,
        DATASET2=DATA_HR,
        T_ITERS=args.t_iters,
        f_LR=args.f_lr,
        T_LR=args.t_lr,
        BATCH_SIZE=args.batch_size
    )

    assert torch.cuda.is_available()
    torch.cuda.set_device(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    elif len(os.listdir(OUTPUT_PATH)) != 0:
        raise AssertionError(f"{OUTPUT_PATH} is not an empty directory. Training is impossible")

    train_sampler = load_train_sampler(batch_size=args.batch_size, device=device)
    val_sampler = load_val_sampler(batch_size=args.batch_size, device=device)

    torch.cuda.empty_cache()
    gc.collect()

    f = ResNet_D(args.img_size, nc=3).cuda()
    f.apply(weights_init_D)

    T = U2NET(in_ch=3, out_ch=3).cuda() if args.model == "unet2" else UNet(3, 3, base_factor=48).cuda()

    torch.manual_seed(0xBADBEEF)
    np.random.seed(0xBADBEEF)

    X_fixed, Y_fixed = train_sampler.sample_paired(10)
    X_test_fixed, Y_test_fixed = val_sampler.sample_paired(10)

    T_opt = torch.optim.Adam(T.parameters(), lr=args.t_lr, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=args.f_lr, weight_decay=1e-10)

    loss_fn = VGGLoss().cuda() if args.cost == "vgg" else F.mse_loss
    metrics = Metrics(device)

    wandb_mode = "disabled" if args.dry_run else "online"
    with wandb.init(name=EXP_NAME, project="SuperResWithNOT", config=config, mode=wandb_mode):
        print("Start running the train loop")
        for step in trange(1, args.max_steps, total=args.max_steps, desc="Training"):
            # T optimization
            unfreeze(T)
            freeze(f)
            for _ in trange(args.t_iters, desc="Running t iters ...", leave=False):
                T_opt.zero_grad()
                if args.supervised:
                    X, Y = train_sampler.sample_paired(args.batch_size)
                    T_X = T(X)
                    T_loss = loss_fn(Y, T_X).mean() - f(T_X).mean()
                else:
                    X = train_sampler.sample_x(args.batch_size)
                    T_X = T(X)
                    T_loss = loss_fn(X, T_X).mean() - f(T_X).mean()

                T_loss.backward()
                T_opt.step()
            wandb.log({f'T_loss': T_loss}, step=step)
            
            del T_loss, T_X, X
            gc.collect()
            torch.cuda.empty_cache()

            # f optimization
            freeze(T)
            unfreeze(f)

            if args.supervised:
                X, Y = train_sampler.sample_paired(args.batch_size)
            else:
                X = train_sampler.sample_x(args.batch_size)
                Y = train_sampler.sample_y(args.batch_size)

            with torch.no_grad():
                T_X = T(X)
            f_opt.zero_grad()
            f_loss = f(T_X).mean() - f(Y).mean()
            f_loss.backward()
            f_opt.step()
            wandb.log({f'f_loss': f_loss.item()}, step=step)
            # del f_loss, Y, X, T_X
            # gc.collect()
            # torch.cuda.empty_cache()
            
            if step % args.metric_freq == 0:
                for _ in trange(args.test_ites, desc="Runing test iterations ...", leave=False, total=args.test_ites):
                    
                    with torch.no_grad():
                        X, Y = val_sampler.sample_paired(args.test_batch)
                        T_X = T(X)
                    
                    real = tensor2image(Y, (-1.0, 1.0), device)
                    fake = tensor2image(T_X, (-1.0, 1.0), device)
                    metrics.update(fake, real)
                
                metrics_dict = metrics.compute(reset=True)
                wandb.log(metrics_dict)

            if step % args.plot_interval == 0:
                clear_output(wait=True)

                fig, _ = plot_images(X_fixed, Y_fixed, T)
                wandb.log({'Fixed Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

                X_random, Y_random = train_sampler.sample_paired(args.batch_size)
                fig, _ = plot_images(X_random, Y_random, T)
                wandb.log({'Random Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

                fig, _ = plot_images(X_test_fixed.cuda(), Y_test_fixed.cuda(), T)
                wandb.log({'Fixed Test Images': [wandb.Image(fig2img(fig))]}, step=step)
                plt.close(fig)

            if step % args.cpkt_interval == 0:
                freeze(T)
                torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))

            # gc.collect()
            # torch.cuda.empty_cache()
            
    freeze(T)
    torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'last_step_ckpt.pt'))


if __name__ == '__main__':
    main()
