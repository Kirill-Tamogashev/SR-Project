# Super Resolution on weather images

## Workflow
### Repo Installation
Repo is installed using the following command:
```shell
pip install -e .
```
Configuration details can be found in `pyproject.toml`. It is 
recommended to use docker to run evaluations and experiments. 
Appropriate docker can be build using `Dockerfile` with the 
following command:
```shell
docker build -t project-sr-docker .
```

### Adding submodules
This repo heavily relies on the submodules – other repositories with 
the particular algorithm. Currently, the submodules include `SR3-Diffusion`,
`Pix2Pix GAN`, `Neural Optimal Transport`, `UNet`. To install new submodule
run 
```shell
git submodule add https://github.com/path/to/NewModule.git ./submodules/NewModule
```
More information can be found [here](https://git-scm.com/book/en/v2/Git-Tools-Submodules).
### Model checkpoints
Info to be added soon ...

## Training and evaluation

This repo provides code for training and evaluating models, presented in 
submodules folder. Next we describe how to run `train` and `eval` scripts
for every available model.

### UNet-based regression
Train:
```shell
unet_train -n UNET_MODEL_NAME -f FINETUNE_MODEL_NAME 
```
### Pix2Pix regression
### SR3-Diffusion regression
### Neural Optimal Transport regression

