import torch
import submodules.ddpm_sr3.data as Data
import submodules.ddpm_sr3.model as Model
import argparse
import logging
import submodules.ddpm_sr3.core.logger as Logger
import submodules.ddpm_sr3.core.metrics as Metrics
import numpy as np
import wandb

from tqdm import tqdm

from ml_collections import ConfigDict

from src.constants import BaseCheckpoint


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default='src/sr_ddpm/sr_config.yaml')
    parser.add_argument('-p', '--phase', choices=('train', 'val'), default='train')
    parser.add_argument('-b', '--name', default='sr-diffusion')
    parser.add_argument('-g', '--gpu_ids', type=int, default=None)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enable_wandb', action='store_true')
    parser.add_argument('--log_wandb_ckpt', action='store_true')
    parser.add_argument('--log_eval', action='store_true')
    return parser.parse_args()


def configure_dataloader(dataset_opt, phase):
    val_set = Data.create_dataset(dataset_opt, phase)
    return Data.create_dataloader(val_set, dataset_opt, phase)


def train(args):
    opt = ConfigDict(Logger.parse(args))
    opt.name = args.name

    opt.path.checkpoint = BaseCheckpoint.SR3 / opt.name

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO)
    logging.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logging.info('Initial Model Finished')

    train_loader = configure_dataloader(opt["datasets"]["train"], "train")
    val_loader = configure_dataloader(opt["datasets"]["val"], "val")

    # Train
    n_iter = opt.train.n_iter

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])

    valid_data_batch = next(iter(val_loader))
    wandb_config = dict(
        project=opt.project,
        name=opt.name,
        config=dict(opt)
    )

    with wandb.init(**wandb_config) as run:

        current_step, current_epoch = 0, 0
        while current_step < n_iter:

            pbar = tqdm(train_loader, desc=f"Epoch {current_epoch}", total=len(train_loader))
            for train_batch in pbar:

                current_step += 1

                diffusion.feed_data(train_batch)
                diffusion.optimize_parameters()
                run.log(diffusion.get_current_log())

                if current_step % opt.train.val_freq == 0:
                    with torch.no_grad():
                        diffusion.set_new_noise_schedule(
                            opt.model.beta_schedule.val,
                            schedule_phase='val'
                        )
                        diffusion.feed_data(valid_data_batch)
                        logging.info("START SAMPLING FOR VALIDATION")
                        diffusion.test(continous=False)
                        logging.info("END SAMPLING FOR VALIDATION")

                    visuals = diffusion.get_current_visuals()
                    sr_img = Metrics.tensor2img(visuals['SR'])  # uint8
                    hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
                    fake_img = Metrics.tensor2img(visuals['INF'])

                    image_array = np.concatenate((fake_img, sr_img, hr_img), axis=1)
                    wandb_image = wandb.Image(
                        image_array,
                        caption=f"sample at {current_step} steps: FAKE SR | DDPM SR | TRUE HR"
                    )
                    run.log({f"Validation images": wandb_image})

                    diffusion.set_new_noise_schedule(opt.model.beta_schedule.train, 'train')

                if current_step % opt.train.save_checkpoint_freq == 0:
                    diffusion.save_network(current_epoch, current_step)


if __name__ == "__main__":
    args = parse_arguments()
    train(args)
