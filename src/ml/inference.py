#!/usr/bin/env python
"""Load a fastai vision model checkpoint without original data.

This script reconstructs the model architecture used in
``fastai_training.py`` by leveraging fastai's ``create_vision_model``.
Given the architecture name and number of output classes, it builds the
model, loads a PyTorch ``state_dict`` checkpoint and prepares the model
for inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from fastai.vision import models
from fastai.vision.learner import create_timm_model, create_vision_model


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
    """Recreate the fastai vision model for ``arch`` and ``num_classes``."""
    # arch_fn = getattr(models, arch)
    # model = create_vision_model(arch, num_classes, pretrained=False)
    model = create_timm_model(arch, num_classes, pretrained=False)[0]

    return model


def load_checkpoint(
    arch: str, num_classes: int, ckpt: Path, device: str = "cpu"
) -> torch.nn.Module:
    """Instantiate model and load ``ckpt`` state dict."""
    model = build_model(arch, num_classes)
    state_dict = torch.load(ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def get_crops(job, layer_num, data_path='/data/additive/crops_260'):
    """
    Get the crops for a specific job and layer number.

    Args:
        job (str): The job identifier.
        layer_num (int): The layer number to get crops for.
        data_path (str): The path to the directory containing crop images.

    Returns:
        ImageDataLoaders: DataLoaders object containing the crops.
    """
    # Get all image files in the directory
    image_files = list(Path(data_path).glob(f'{job}_{layer_num:05d}*.jpg'))
    return image_files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arch", required=True, help="model architecture, e.g. resnet18")
    p.add_argument("--num-classes", type=int, required=True, help="number of target classes")
    p.add_argument("--checkpoint", type=Path, required=True, help="path to checkpoint file")
    p.add_argument("--device", default="cpu", help="device to load model on")
    args = p.parse_args()

    mm = load_checkpoint(args.arch, args.num_classes, args.checkpoint, args.device)
    mm.eval()
    print(
        f"Loaded {args.arch} model with {args.num_classes} classes from '{args.checkpoint}' on {args.device}"
    )

    # inference

    from torchvision import transforms
    from PIL import Image

    transform = transforms.Compose([
    transforms.Resize(256),  # resize al lado menor = 256 (como fastai)
    transforms.ToTensor(),  # convierte a [0,1] float tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # imagenet_stats
                         std=[0.229, 0.224, 0.225]),
])
    # Define a color for each class (pueden ser cualquier cosa)
    import numpy as np
    colors = np.array([
    [255, 0, 0],    # clase 0 - rojo
    [0, 255, 0],    # clase 1 - verde
    [0, 0, 255],    # clase 2 - azul
    [255, 255, 0],  # clase 3 - amarillo
    [255, 0, 255],  # clase 4 - fucsia
    [0, 255, 255],  # clase 5 - cyan
    [128, 128, 128] # clase 6 - gris
], dtype=np.uint8)
    image_files = get_crops(20193, 2500)

    for image_file in image_files:
        img = Image.open(image_file).convert("RGB")
        img_tensor = transform(img).unsqueeze(0)  # add batch dimension
        with torch.no_grad():
            logits = mm(img_tensor)
            class_idx = torch.argmax(logits, dim=1).item()
            out_class = sorted(['border', 'drop', 'elevation', 'spatter', 'stripe', 'other', 'plain'])[class_idx]
            img.save('input.png')
            print(f"Predicted class for {image_file}: {out_class}")
            spatial_output = mm[0].model.forward_features(img_tensor)
            spatial_output = mm[0].model.fc_norm(spatial_output)
            spatial_output = mm[1](spatial_output.squeeze(0))[4:]
            spatial_output = spatial_output.reshape(16,16,7)
            spatial_output = spatial_output.argmax(axis=-1).cpu().numpy()
            colored = colors[spatial_output].astype(np.uint8)
            Image.fromarray(colored).resize((256,256)).save(f"spatial_output.png")
            breakpoint()
        # print(f"Output for {image_file}: {output}")


if __name__ == "__main__":
    main()