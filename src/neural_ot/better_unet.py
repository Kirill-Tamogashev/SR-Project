import torch

from diffusers import UNet2DModel


class NOTUNet(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
            
        self.unet = UNet2DModel(*args, **kwargs)
        self.time_embedding = _dummy_embedding
        self.time_proj = _dummy_proj
            

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.size(0)
        dummy_t = torch.ones(bs, device=x.device)
        return self.unet(x, dummy_t).sample
        

def _dummy_proj(*args, **kwargs):
    return None
            
def _dummy_embedding(*args, **kwargs):
    return None
