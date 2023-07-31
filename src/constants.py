from enum import Enum
from pathlib import Path


class BaseCheckpoint(Enum):
    UNET: Path = Path("./checkpoints/unet")
    UNET_FINETUNE: Path = Path("./checkpoints/finetune")
    PIX2PIX: Path = Path("./checkpoints/pix2pix")
    NOT: Path = Path("./checkpoints/not")
    SR3: Path = Path("./checkpoints/ddpm_sr3")
