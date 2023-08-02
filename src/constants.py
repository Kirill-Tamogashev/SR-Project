from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseCheckpoint:
    UNET: Path = Path("./checkpoints/unet")
    UNET_FINETUNE: Path = Path("./checkpoints/finetune-unet")
    PIX2PIX: Path = Path("./checkpoints/pix2pix")
    NOT: Path = Path("./checkpoints/not")
    SR3: Path = Path("./checkpoints/ddpm_sr3")
