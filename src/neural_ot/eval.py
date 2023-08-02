from argparse import ArgumentParser

import torch
from tqdm import tqdm
import os

from submodules.torch_not.src.unet import UNet

from src.constants import BaseCheckpoint
from src.miscellaneous.metrics import Metrics
from src.miscellaneous.utils import tensor2image
from src.neural_ot.unet2 import U2NET
from src.neural_ot.data_samplers import load_test_dataloader

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_ckpt(model: str, device):
    model_path = BaseCheckpoint.NOT / model
    latest_ckpt_path = max(model_path.glob("*.pt"), key=lambda x: x.stat().st_ctime)
    return torch.load(latest_ckpt_path, map_location=device)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--net", type=str, default="unet", choices={"unet", "unet2"})
    parser.add_argument("--model", type=str, required=True, choices=os.listdir(BaseCheckpoint.NOT))
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


@torch.no_grad()
def eval_not(args):
    device = torch.device(f"cuda:{args.device}")
    ckpt = load_ckpt(args.model, device)

    model_T = U2NET(3, 3) if args.net == "unet2" else UNet(3, 3, base_factor=48)
    model_T.load_state_dict(ckpt)
    model_T.to(device)
    model_T.eval()

    test_loader = load_test_dataloader(args.batch_size)
    metrics = Metrics(device)

    for batch in tqdm(test_loader):
        X = batch["SR"].to(device)
        Y = batch["HR"].to(device)

        T_X = model_T(X)

        Y = tensor2image(Y, (-1.0, 1.0), device)
        T_X = tensor2image(T_X, (-1.0, 1.0), device)

        metrics.update(real=Y, fake=T_X)
    metrics.print()


def main():
    args = parse_args()
    eval_not(args)   


if __name__ == '__main__':
    main()
