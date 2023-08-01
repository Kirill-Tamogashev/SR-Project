import typing as tp
from pathlib import Path
import os

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FinetuneModel(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hidden: int = 64,
            use_skip_connection: bool = True
    ):
        super().__init__()
        self._use_skip_connection = use_skip_connection
        self._out_channels = out_channels
        self._model = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Conv2d(hidden, out_channels, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        h = self._model(x)
        if self._use_skip_connection:
            if self._out_channels == 3:
                residual = x
            elif self._out_channels == 2:
                residual = x[:, 1:, :, :]
            else:
                residual = x[:, :1, :, :]
            h = h + residual
        return h


class RegressionSR(nn.Module):
    def __init__(
            self,
            params,
            models: tp.List,
            model_chkpt_dir: Path,
            loss_fn: tp.Callable,
            use_scheduler: bool = True
    ):
        super().__init__()

        self.model_ckpt_dir = model_chkpt_dir
        self.loss_fn = loss_fn

        self.models = models
        self.optimizers = [
            torch.optim.Adam(
                model.parameters(),
                lr=params.training.learning_rate,
                weight_decay=params.training.weight_decay
            ) for model in self.models
        ]

        self.schedulers = [
            ReduceLROnPlateau(opt, 'min', patience=5) if use_scheduler else None
            for opt in self.optimizers
        ]
        
    def to(self, device):
        for model in self.models:
            model.to(device)
            
    def train(self):
        for model in self.models:
            model.train()
    
    def eval(self):
        for model in self.models:
            model.eval()

    def infer(self, x):
        preds = self(x)
        return torch.cat(preds, dim=1) if isinstance(preds, tuple) else preds[0]

    def forward(self, x):
        return [model(x) for model in self.models]
    
    def compute_losses(self, preds, targets):
        targets = targets.split([1, 2], dim=1) if len(preds) == 2 else (targets, )
        return [self.loss_fn(x, trg) for x, trg in zip(preds, targets)]

    @torch.no_grad()
    def val_step(self, inputs, targets):
        preds = self(inputs)
        losses = self.compute_losses(preds, targets)

        out = torch.cat(preds, dim=1) if isinstance(preds, tuple) else preds[0]
        return out, losses

    def train_step(self, inputs, targets):
        preds = self(inputs)
        losses = self.compute_losses(preds, targets)

        for opt, scheduler, loss in zip(self.optimizers, self.schedulers, losses):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if scheduler is not None:
                scheduler.step(loss)

        out = torch.cat(preds, dim=1) if isinstance(preds, tuple) else preds[0]
        return out, losses

    def save_checkpoint(
            self,
            epoch: int,
            epochs: int,
            prefix: str,
    ):
        ckpt_stat_dict = {
            "MODEL_STATE": [model.state_dict() for model in self.models],
            "OPTIMIZER_STATE": [opt.state_dict() for opt in self.optimizers],
        }

        if (epoch and epoch % 5 == 0) or epoch == epochs:
            path = self.model_ckpt_dir / f'{prefix}-milestone-{epoch}.pth'
        else:
            path = self.model_ckpt_dir / f'{prefix}-checkpoint-epoch-{epoch}.pth'
        torch.save(ckpt_stat_dict, path.as_posix())

        if epoch - 3 > 0 and (epoch - 3) % 5 != 0:
            path_to_remove = self.model_ckpt_dir / f'{prefix}-checkpoint-epoch-{epoch - 3}.pth'
            os.remove(path_to_remove.as_posix())

    def load_checkpoint(self, device):
        latest_ckpt_path = max(self.model_ckpt_dir.glob("*.pth"), key=lambda x: x.stat().st_ctime)
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model_states = checkpoint["MODEL_STATE"]

        for model, state_dict in zip(self.models, model_states):
            model.load_state_dict(state_dict)
