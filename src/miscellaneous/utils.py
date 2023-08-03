from typing import Any, Tuple, Union
from pathlib import Path

import torch


def tensor2image(
        tensor:     torch.Tensor,
        min_max:    Tuple[float, float] = (-1.0, 1.0),
        device:     Union[str, torch.device] = "cuda:0"
) -> torch.Tensor:
    min_, max_ = min_max
    tensor = tensor.sub(min_).div(max_ - min_).mul(255).byte()
    return tensor.to(device)


def load_ckpt(
    model_path: Path, 
    device: Union[str, torch.device], 
    pattern: str = "*.pt",
) -> Any:
    latest_ckpt_path = max(model_path.glob(pattern), key=lambda x: x.stat().st_ctime)
    return torch.load(latest_ckpt_path, map_location=device)
