import argparse
import logging
from pathlib import Path
import torch

from tqdm import tqdm
import os

from submodules.torch_unet.unet import UNet

from src.unet.model import RegressionSR, FinetuneModel
from src.unet.utils import (
    normalize,
    denormalize,
    load_params,
    configure_dataloader,
)
from src.constants import BaseCheckpoint
from src.miscellaneous.metrics import Metrics

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@torch.no_grad()
def eval_unet(params) -> None:
    device = torch.device(f"cuda:{params.training.device}" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    test_loader = configure_dataloader(
        low_res_path=params.data.test.lr,
        high_res_path=params.data.test.hr,
        batch_size=params.training.batch_size,
        shuffle=True
    )

    unet_ckpt_dir = BaseCheckpoint.UNET / params.unet_model.name

    logging.info(f"Loading {params.unet_model.name} as a base UNet model.")
    unet_nets = [UNet(3, 1, True), UNet(3, 2, True)] \
        if params.unet_model.channelwise else [UNet(3, 3, True)]
    unet_model = RegressionSR(
        params=params,
        loss_fn=None,
        models=unet_nets,
        model_chkpt_dir=unet_ckpt_dir
    )
    unet_model.load_checkpoint(device)
    unet_model.to(device)
    unet_model.eval()
    logging.info("UNet model configured.")

    use_finetune = params.finetune_model.finetune
    if use_finetune:
        logging.info(f"Loading {params.finetune_model.name} as a base UNet model.")
        finetune_ckpt_dir = BaseCheckpoint.UNET_FINETUNE / params.finetune_model.name

        finetune_nets = [FinetuneModel(3, 1, 64, True), FinetuneModel(3, 2, 64, True)] \
            if params.finetune_model.channelwise else [FinetuneModel(3, 3, 64, True)]

        finetune_model = RegressionSR(
            params=params,
            loss_fn=None,
            models=finetune_nets,
            model_chkpt_dir=finetune_ckpt_dir
        )
        finetune_model.to(device)
        finetune_model.eval()
        logging.info("Finetune model configured")
    else:
        logging.info("No finetune model")

    metrics = Metrics(device)

    logging.info("Begin evaluation.")
    for batch in tqdm(test_loader):

        sr_input = batch["SR"].to(device=device)
        hr_true = batch["HR"].to(device=device).byte()

        sr_normed, sr_min_values, sr_max_values = normalize(sr_input)

        sr_unet_out = unet_model.infer(sr_normed)
        if use_finetune:
            sr_pred = finetune_model.infer(sr_unet_out)
        else:
            sr_pred = sr_unet_out

        hr_pred = denormalize(sr_pred, sr_min_values, sr_max_values)
        metrics.update(real=hr_true, fake=hr_pred)

    logging.info("Evaluation finished, printing metrics")
    metrics.print()


def parse_arguments():
    parser = argparse.ArgumentParser(description='UNet for SR task')
    parser.add_argument('--unet-name', '-n', type=str, default="unet-sr", 
                        help="Name of the UNet model", choices=os.listdir(BaseCheckpoint.UNET))
    parser.add_argument('--finetune-name', '-f', type=str, default="finetune-sr", 
                        help="Name of the Finetune model", 
                        choices=os.listdir(BaseCheckpoint.UNET_FINETUNE))
    parser.add_argument('--gpu', '-g', type=str, default=None, help="device number")
    parser.add_argument('--params', '-p', type=Path, default="./src/unet/params.yaml")
    parser.add_argument('--finetune', action="store_true", help="Use finetune")
    parser.add_argument('--batch_size', "-b", type=int, default=32, 
                        help="Batch size used for eval")
    return parser.parse_args()


def main():
    train_args = parse_arguments()
    params = load_params(train_args.params)

    params.unet_model.name = train_args.unet_name
    params.finetune_model.name = train_args.finetune_name
    params.training.device = train_args.gpu
    params.finetune_model.finetune = train_args.finetune
    params.training.batch_size = train_args.batch_size

    eval_unet(params)


if __name__ == '__main__':
    main()
