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
## to-do
- make the router show both options if task is not set in config and show only the selected task if it is, and fail if going directly to one of the tasks if it's not the selected task, and if it is the right one and no task was set, set it 
- make using the keyboard easier (should work everywhere in the app)
- when removing a class show warning, remove the annotations for that class, and reset the model
- remove the live accuracy things in the seg frontend 
