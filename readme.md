# Specification

A human-in-the-loop image annotation system with continuous training.

## Install
You need to install:
- the packages in requirements.txt (you can install `uv` and do `uv venv --python 3.13` and then `uv pip install -r requirements.txt`)
- dinov3
    - by cloning the repo `cd src/ml/ && git clone https://github.com/facebookresearch/dinov3`
    - and downloading the weights you wanna use to `src/ml/weights/` (we support small and large for now)

## Run
edit db_init.py to configure what images you will load

you must launch both

webapp (local):
```
gunicorn src.backend.main:app --bind 127.0.0.1:8001 --reload
```
AI training (classification):
```
python -m src.ml.fastai_training --arch vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k
```
AI training (segmentation):
``` 
python -m src.ml.dinov3_training 
```