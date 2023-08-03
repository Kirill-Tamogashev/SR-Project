from pathlib import Path

from ml_collections import ConfigDict
# import os
import yaml

from src.constants import BaseCheckpoint


def load_params(params_file: Path) -> ConfigDict:
    with params_file.open() as file:
        params: dict = yaml.safe_load(file)
    return ConfigDict(params)


def configure_params(args):
    params = load_params(args.config)
    
    params.name = args.name
    params.phase = args.phase
    params.gpu_ids = [args.device]
    
    params.path.experiments_root = BaseCheckpoint.SR3 / args.name
    
    return params
