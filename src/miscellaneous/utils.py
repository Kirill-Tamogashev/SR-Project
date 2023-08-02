from typing import Tuple, Union

import torch


def tensor2image(
        tensor: torch.Tensor,
        min_max: Tuple[float, float] = (-1.0, 1.0),
        device: Union[str, torch.device] = "cuda"
) -> torch.Tensor:
    min_, max_ = min_max
    tensor = tensor.sub(min_).div(max_ - min_).mul(55).byte()
    return tensor.to(device)
