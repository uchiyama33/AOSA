import os
from glob import glob
import json
import re

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy


def try_make_dir(d):
    if not os.path.isdir(d):
        os.makedirs(d)


def normalize_heatmap(heatmap, base_val, model_type=None):
    if type(base_val) == torch.Tensor:
        base_val = base_val.detach().cpu().numpy()
    if type(heatmap) == torch.Tensor:
        heatmap = heatmap.detach().cpu()
    norm_heatmap = heatmap - base_val

    if model_type is not None:  # dist to sim
        if model_type == "affine":
            norm_heatmap *= -1

    min_norm, max_norm = (
        norm_heatmap.min(),
        norm_heatmap.max(),
    )
    scale = np.max(np.abs([max_norm, min_norm]))

    if scale > 1e-10:
        norm_heatmap = norm_heatmap / scale * 127.5
        norm_heatmap += 127.5
    return norm_heatmap


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_class_labels_map(ground_truth_path):
    with ground_truth_path.open('r') as f:
        data = json.load(f)
    class_labels_map = get_class_labels(data)
    return class_labels_map


def video_to_html(video_orgimg, save_path=None):
    fig = plt.figure()
    plt.axis("off")
    ims = []
    for orgimg in video_orgimg:
        im = plt.imshow(orgimg, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims)
    plt.close(fig)

    if save_path:
        ani.save(save_path)

    return ani.to_jshtml()


def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts
