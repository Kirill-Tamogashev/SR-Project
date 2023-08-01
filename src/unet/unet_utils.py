import typing as tp
from pathlib import Path

import yaml
import wandb
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from ml_collections import ConfigDict

from src.data.dataset import Dataset


def normalize(image) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_values = image.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0]
    max_values = image.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]

    normalized_image = (image - min_values) / (max_values - min_values)
    return normalized_image, min_values, max_values


def denormalize(image, min_values, max_values):
    image = image * (max_values - min_values) + min_values
    return torch.clip(image, min=0, max=255)


def cls_name(cls):
    return cls.__class__.__name__


def make_image(image_list):
    titles = ("SR Image", "HR Image", "Bicubic")
    fig, ax = plt.subplots(3, 10, figsize=(90, 30))

    for row, image_batch in enumerate(image_list):
        title = titles[row]
        for col, image in enumerate(image_batch):
            image = torch.clip(image.to(torch.uint8), min=0, max=255)
            image = image.permute(1, 2, 0).cpu().numpy()
            ax[row, col].imshow(image)
            ax[row, col].set_title(f"{title} | image {col}", fontsize=45)
            ax[row, col].axis("off")
    return fig


def load_params(params_file: Path) -> ConfigDict:
    with params_file.open() as file:
        params: dict = yaml.safe_load(file)
    return ConfigDict(params)


def configure_dataloader(
        low_res_path: str,
        high_res_path: str,
        batch_size: int,
        shuffle: bool = False
):
    dataset = Dataset(
        lr_path=low_res_path,
        hr_path=high_res_path,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )


def log_data(
        run,
        sr_img,
        hr_img,
        sr_unet,
        losses,
        global_step:    int,
        epoch:          int,
        stage:          str = "train",
    ):
    if len(losses) == 1:
        loss_dict = {f"{stage}/loss": losses[0]}
    else:
        loss_dict = {
            f"{stage}/3 -> 1 loss": losses[0],
            f"{stage}/3 -> 2 loss": losses[1],
        }
    run.log(loss_dict)

    if global_step % 1 == 0:
        total, channelwise = compute_prod_metrics(sr_unet, hr_img)
        with torch.no_grad():
            image = make_image([sr_unet[:10], hr_img[:10], sr_img[:10]])
        wandb_image = wandb.Image(image, caption=f"epoch {epoch}, global step={global_step}")
        run.log(
            {
                f"prod metrics/{stage}/total MSE": total.item(),
                f"prod metrics/{stage}/MSE: channel 0": channelwise[0],
                f"prod metrics/{stage}/MSE: channel 1": channelwise[1],
                f"prod metrics/{stage}/MSE: channel 2": channelwise[2],
                f"images {stage}/images": wandb_image
            }
        )
        plt.close(image)


def compute_prod_metrics(batch_sr, batch_hr) -> tp.Tuple[torch.Tensor, torch.Tensor]:
    total_mse = (batch_sr - batch_hr).pow(2).mean()
    channelwise = (batch_sr - batch_hr).pow(2).mean(dim=(0, 2, 3))
    return total_mse, channelwise
