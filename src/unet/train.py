import argparse
import logging
from pathlib import Path
import torch

import wandb
from tqdm import tqdm

from submodules.torch_unet.unet import UNet

from src.unet.model import RegressionSR, FinetuneModel
from src.unet.utils import (
    normalize,
    denormalize,
    load_params,
    configure_dataloader,
    log_data,
    cls_name
)
from src.miscellaneous.losses import configure_loss
from src.constants import BaseCheckpoint

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 


def train(params, run) -> None:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device(f"cuda:{params.training.device}" \
        if torch.cuda.is_available() else "cpu")

    train_loader = configure_dataloader(
        low_res_path=params.data.train.lr,
        high_res_path=params.data.train.hr,
        batch_size=params.training.batch_size,
        shuffle=True
    )
    val_loader = configure_dataloader(
        low_res_path=params.data.val.lr,
        high_res_path=params.data.val.hr,
        batch_size=10,
        shuffle=True
    )
    logging.info("Dataloaders configured.")
    val_batch = next(iter(val_loader))

    loss_fn = configure_loss(
        params.training.criterion,
        params.training.loss_args,
        device
    )
    logging.info(f"Loss configured. Using {cls_name(loss_fn)}")
    
    unet_ckpt_dir = BaseCheckpoint.UNET / params.unet_model.name
    unet_ckpt_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Base model checkpoints path {unet_ckpt_dir.as_posix()}")

    unet_nets = [UNet(3, 1, True), UNet(3, 2, True)] \
        if params.unet_model.channelwise else [UNet(3, 3, True)]
    unet_model = RegressionSR(
        params=params,
        loss_fn=loss_fn,
        models=unet_nets,
        model_chkpt_dir=unet_ckpt_dir
    )
    unet_model.to(device)
    unet_model.train()
    logging.info("UNet model configured.")
    logging.info(f"Train channelwise: {params.unet_model.channelwise}.")

    use_finetune = params.finetune_model.finetune
    
    if use_finetune:
        logging.info("Configuring finetune model.")
        logging.info(f"Loading {params.unet_model.name} as a base UNet model.")
        unet_model.load_checkpoint(device)
        unet_model.eval()

        finetune_ckpt_dir = BaseCheckpoint.UNET_FINETUNE / params.finetune_model.name
        finetune_ckpt_dir.mkdir(parents=True, exist_ok=True)

        finetune_nets = [FinetuneModel(3, 1, 64, True), FinetuneModel(3, 2, 64, True)] \
            if params.finetune_model.channelwise else [FinetuneModel(3, 3, 64, True)]
        finetune_model = RegressionSR(
            params=params,
            loss_fn=loss_fn,
            models=finetune_nets,
            model_chkpt_dir=finetune_ckpt_dir
        )
        finetune_model.to(device)
        finetune_model.train()
        logging.info("Finetune model configured")
        logging.info(f"Finetune channelwise: {params.finetune_model.channelwise}")
    else:
        finetune_model = None
        logging.info("No finetune model")

    global_step = 0
    epochs = params.training.n_epochs
    logging.info("Begin training.")
    for epoch in range(1, epochs + 1):
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                sr_input = batch["SR"].to(device=device)
                hr_true = batch["HR"].to(device=device)

                sr_normed, sr_min_values, sr_max_values = normalize(sr_input)
                hr_normed, *_ = normalize(hr_true)

                if use_finetune:
                    with torch.no_grad():
                        sr_pred = unet_model.infer(sr_normed)

                    finetune_model.train()
                    sr_unet, losses = finetune_model.train_step(sr_pred, hr_normed)
                else:
                    unet_model.train()
                    sr_unet, losses = unet_model.train_step(sr_normed, hr_normed)
                
                hr_pred = denormalize(sr_unet, sr_min_values, sr_max_values)
                if run is not None:
                    log_data(run, sr_input, hr_true, hr_pred, losses, global_step, epoch, "train")

                global_step += 1
                pbar.update()

            with torch.no_grad():

                sr_val = val_batch["SR"].to(device)
                hr_val = val_batch["HR"].to(device)

                sr_normed, sr_min_values, sr_max_values = normalize(sr_val)
                hr_normed, *_ = normalize(hr_val)

                if params.finetune:
                    with torch.no_grad():
                        sr_pred = unet_model.infer(sr_normed)

                    finetune_model.eval()
                    sr_unet, losses = finetune_model.val_step(sr_pred, hr_normed)
                    finetune_model.save_checkpoint(epoch, epochs, prefix="finetune")
                    finetune_model.train()
                else:
                    unet_model.eval()
                    sr_unet, losses = unet_model.val_step(sr_normed, hr_normed)
                    unet_model.save_checkpoint(epoch, epochs, prefix="unet")
                    unet_model.train()

                hr_pred = denormalize(sr_pred, sr_min_values, sr_max_values)

                if run is not None:
                    log_data(run, sr_input, hr_true, hr_pred, losses, global_step, epoch, "val")


def parse_arguments():
    parser = argparse.ArgumentParser(description='UNet for SR task')
    parser.add_argument('--unet-name', '-n', type=str, default="unet-sr", help="Name of the UNet model")
    parser.add_argument('--finetune-name', '-f', type=str, default="finetune-sr", help="Name of the Finetune model")
    parser.add_argument('--project', type=str, default="U-Net-SR", help="Name of the project")
    parser.add_argument('--gpu', '-g', type=str, default=None, help="device number")
    parser.add_argument('--params', '-p', type=Path, default="./src/unet/unet_params.yaml")
    parser.add_argument('--finetune', action="store_true", help="Use finetune")
    parser.add_argument('--wandb', action="store_true", help="Use wandb")
    return parser.parse_args()


def main():
    train_args = parse_arguments()
    params = load_params(train_args.params)

    params.unet_model.name = train_args.unet_name
    params.finetune_model.name = train_args.finetune_name
    params.project = train_args.project
    params.training.device = train_args.gpu
    params.finetune_model.finetune = train_args.finetune

    if train_args.wandb:
        wandb_config = dict(
            name=params.name,
            project=params.project,
            config=dict(params)
        )
        with wandb.init(**wandb_config) as run:
            train(params, run)
    else:
        train(params, None)


if __name__ == '__main__':
    main()
