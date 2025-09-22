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

and the AI, which is one of the following:
- classification training (timm arch):
    ```
    python -m src.ml.fastai_training --arch vit_medium_patch16_reg4_gap_256.sbb_in12k_ft_in1k
    ```
- classification training (frozen dinov3 + linear):
    ```
    python -m src.ml.dinov3_classification
    ```
- segmentation training (frozen dinov3 + linear):
    ``` 
    python -m src.ml.dinov3_training 
    ```

## to-do
- check cls frontend works
- make the router show both options if task is not set in config and show only the selected task if it is, and fail if going directly (using url) to one of the tasks if it's not the selected task, and if it is the right one and no task was set, set it. essentially, if the task is not set and we hit the router, show both tasks. once we move to a task, if the task is unset, we set it. if one moves to a task that was set and is different, show an alert that says something like 'the current task is X, you can't change tasks, if you want to do another task, reset the app' (resetting means rm / renaming current session dir and relaunching the apps)
- make using the keyboard easier (should work everywhere in the app)
- when removing a class show warning, remove the annotations for that class, and reset the model
- remove the live accuracy things in the seg frontend 
- add next image strategy (random, sequential, select class) for seg frontend
- show image id in frontend
- allow the user to jump to a specific image id
