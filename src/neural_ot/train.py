from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os

import wandb
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

import torch
import torch.nn.functional as F

from submodules.torch_not.src.resnet2 import ResNet_D
from submodules.torch_not.src.unet import UNet
from submodules.torch_not.src.tools import unfreeze, freeze
from submodules.torch_not.src.tools import weights_init_D
from submodules.torch_not.src.plotters import plot_images
from submodules.torch_not.src.tools import fig2img

from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image
from src.miscellaneous.losses import VGGPerceptualLoss as VGGLoss
from src.neural_ot.unet2 import U2NET
from src.neural_ot.better_unet import NOTUNet
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
    
    parser.add_argument("--batch-size",     help="Batch size for traning", type=int, default=64)
    parser.add_argument("--test-batch",     help="Batch size for testing", type=int, default=25)
    parser.add_argument("--max-steps",      help="Number of max steps", type=int, default=10_001)
    parser.add_argument("--test-iters",      help="Numbet of testing iterations", type=int, default=200)
    parser.add_argument("--t-iters",        help="Number of iterations for T", type=int, default=10)
    parser.add_argument("--img-size",       help="Size of image", type=int, default=128)
    parser.add_argument("--f-lr",           help="Learning rate for f", type=float, default=1e-4)
    parser.add_argument("--t-lr",           help="Learning rate for T", type=float, default=1e-4)
    parser.add_argument("--gpu-id",         help="Device for traning", type=str, default=None)
    
    parser.add_argument("--cpkt-freq",      help="Frequency of checkpointing", type=int, default=2000)
    parser.add_argument("--plot-freq",      help="Frequency of plotting", type=int, default=1000)
    parser.add_argument("--metric-freq",    help="Frequency of metric printing", type=int, default=200)
    
    parser.add_argument("--cost",           help="Cost function C", type=str, default="mse", choices=("mse", "vgg"))
    parser.add_argument("--model",          help="Model type", type=str, default="unet", choices=("unet", "unet", "unet_attn"))
    
    parser.add_argument("--dry-run",        action="store_true", help="Disable wandb")
    parser.add_argument("--supervised",     action="store_true", help="Enable superwised OT")
    return parser.parse_args()


def configure_unet(model_name: str):
    if model_name == "unet":
        return UNet(3, 3, base_factor=48)
    elif model_name == "unet2":
        return U2NET(3, 3)
    elif model_name == "unet_attn":
        unet_config = dict(
            sample_size = 128,
            in_channels = 3,
            out_channels = 3,
            center_input_sample = False,
            time_embedding_type = "positional",
            freq_shift = 0,
            flip_sin_to_cos = True,
            down_block_types = [
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D"
            ],
            up_block_types = [
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ],
            block_out_channels = [128, 128, 256, 256, 512, 512],
            layers_per_block = 2,
            mid_block_scale_factor = 1,
            downsample_padding = 1,
            act_fn = 'silu',
            attention_head_dim = 8,
            norm_num_groups = 32,
            norm_eps = 1e-05,
            resnet_time_scale_shift = 'default',
            add_attention = True,
            class_embed_type = None,
            num_class_embeds = None
        )
        return NOTUNet(**unet_config)
    else:
        raise ValueError(f"Unknown model name {model_name}")


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

    f = ResNet_D(args.img_size, nc=3).cuda()
    f.apply(weights_init_D)

    T = configure_unet(args.model)
    T.cuda()
    T.train()

    torch.manual_seed(0xBADBEEF)
    np.random.seed(0xBADBEEF)

    X_fixed, Y_fixed = train_sampler.sample_paired(10)
    X_test_fixed, Y_test_fixed = val_sampler.sample_paired(10)

    T_opt = torch.optim.Adam(T.parameters(), lr=args.t_lr, weight_decay=1e-10)
    f_opt = torch.optim.Adam(f.parameters(), lr=args.f_lr, weight_decay=1e-10)

    loss_fn = VGGLoss().cuda() if args.cost == "vgg" else F.mse_loss
    metrics = Metrics(device)

    wandb_mode = "disabled" if args.dry_run else "online"
    with wandb.init(name=EXP_NAME, project="SuperResWithNOT", config=config, mode=wandb_mode) as run:
        print("Start running the train loop")
        for step in trange(1, args.max_steps, total=args.max_steps, desc="Training"):
            # T optimization
            unfreeze(T)
            freeze(f)
            for _ in trange(args.t_iters, desc="Running t iterations ...", leave=False):
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
            run.log({f'Train/T_loss': T_loss}, step=step)

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
            run.log({f'Train/f_loss': f_loss.item()}, step=step)
            
            if step % args.metric_freq == 0:
                for _ in trange(args.test_iters, desc="Runing test iterations ...", leave=False, total=args.test_iters):
                    
                    with torch.no_grad():
                        X, Y = val_sampler.sample_paired(args.test_batch)
                        T_X = T(X)
                    
                    real = tensor2image(Y, device=device)
                    fake = tensor2image(T_X, device=device)
                    metrics.update(fake, real)
                
                metrics_dict = metrics.compute(reset=True)
                metrics_dict = {
                    f"Validation/{name}": metric for name, metric in metrics_dict.items()
                }
                run.log(metrics_dict, step=step)

            if step % args.plot_freq == 0:
                fig, _ = plot_images(X_fixed, Y_fixed, T)
                run.log({"Images/Fixed train Images": wandb.Image(fig2img(fig))}, step=step)
                plt.close(fig)

                X_random, Y_random = train_sampler.sample_paired(args.batch_size)
                fig, _ = plot_images(X_random, Y_random, T)
                run.log({"Images/Random train Images": wandb.Image(fig2img(fig))}, step=step)
                plt.close(fig)
                
                X_random_val, Y_random_val = val_sampler.sample_paired(args.batch_size)
                fig, _ = plot_images(X_random_val, Y_random_val, T)
                run.log({"Images/Random validation Images": wandb.Image(fig2img(fig))}, step=step)
                plt.close(fig)
                
                fig, _ = plot_images(X_test_fixed, Y_test_fixed, T)
                run.log({"Images/Fixed validation Images": wandb.Image(fig2img(fig))}, step=step)
                plt.close(fig)

            if step % args.cpkt_freq == 0:
                freeze(T)
                torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'{SEED}_{step}.pt'))

    freeze(T)
    torch.save(T.state_dict(), os.path.join(OUTPUT_PATH, f'last_step_ckpt.pt'))


if __name__ == '__main__':
    main()
