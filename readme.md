# Specification

A human-in-the-loop image annotation system with continuous training.

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