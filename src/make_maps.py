from org3dresnet.main import (get_normalize_method, get_inference_utils,
                              get_opt, generate_model, resume_model)
from org3dresnet.model import (generate_model, load_pretrained_model, make_data_parallel,
                               get_fine_tuning_parameters)
import org3dresnet

import torch
import numpy as np
import cv2
from torch.backends import cudnn
import torch.nn.functional as F
import torchvision
from IPython.display import HTML
from torchvision.transforms.transforms import Normalize, ToPILImage
from torchvision.transforms import transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from captum.attr import LayerGradCam, DeepLift, Occlusion
from captum.attr import visualization as viz

import os
import sys
import json
import argparse
from argparse import Namespace
from pathlib import Path
from time import time
from tqdm import tqdm

from utils.visualization import visualize
from utils.utils import video_to_html, normalize_heatmap
from utils.sensitivity_analysis import OcclusionSensitivityMap3D as OSM
from utils.approx_sensitivity_analysis import (
    ApproxOcclusionSensitivityMap3D as AOSM,
)


def make(args):
    with open(args.opt_path, "r") as f:
        model_opt = json.load(f)
    model_opt = Namespace(**model_opt)

    model_opt.device = torch.device('cpu' if model_opt.no_cuda else 'cuda')
    if not model_opt.no_cuda:
        cudnn.benchmark = True
    if model_opt.accimage:
        torchvision.set_image_backend('accimage')

    model_opt.ngpus_per_node = torch.cuda.device_count()

    model = generate_model(model_opt)
    model = resume_model(model_opt.resume_path, model_opt.arch, model)
    model_feat = create_feature_extractor(model, ["avgpool"])
    model = make_data_parallel(model, model_opt.distributed, model_opt.device)
    model.eval()
    model_feat = make_data_parallel(
        model_feat, model_opt.distributed, model_opt.device)
    model_feat.eval()

    model_opt.inference_batch_size = 1
    for attribute in dir(model_opt):
        if "path" in str(attribute) and getattr(model_opt, str(attribute)) != None:
            setattr(model_opt, str(attribute), Path(
                getattr(model_opt, str(attribute))))
    inference_loader, inference_class_names = get_inference_utils(model_opt)

    class_labels_map = {v.lower(): k for k, v in inference_class_names.items()}
    inputs, targets = iter(inference_loader).__next__()
    video_size = inputs[[0]].shape
    transform = inference_loader.dataset.spatial_transform

    ####
    if args.use_approx:
        osm = AOSM(
            net=model,
            video_size=video_size,
            device=model_opt.device,
            spatial_crop_sizes=args.spatial_crops,
            temporal_crop_sizes=args.temporal_crops,
            spatial_stride=args.spatial_stride,
            temporal_stride=args.temporal_stride,
            transform=transform,
            batchsize=args.batchsize,
            flow_method=args.flow_method,
            N_stack_mask=args.N_stack_mask,
            n_split=0,
            conditional=True,
            n_window=4,
            adjust_method="simple",
        )
    else:
        osm = OSM(
            net=model,
            video_size=video_size,
            device=model_opt.device,
            spatial_crop_sizes=args.spatial_crops,
            temporal_crop_sizes=args.temporal_crops,
            spatial_stride=args.spatial_stride,
            temporal_stride=args.temporal_stride,
            transform=transform,
            batchsize=args.batchsize,
            flow_method=args.flow_method,
            N_stack_mask=args.N_stack_mask,
        )

    ####
    dataset = args.opt_path.split("/")[-3]
    net_type = args.opt_path.split("/")[-2]
    root = os.path.join("/workspace/results/maps", dataset, net_type, args.map_type)
    os.makedirs(root, exist_ok=True)

    with open(os.path.join(root, "args.json"), mode="w") as f:
        json.dump(args.__dict__, f, indent=4)

    log_path = os.path.join(root, "log")
    if os.path.exists(log_path):
        done = torch.load(log_path)
    else:
        done = -1
    for n, (inputs, targets) in tqdm(enumerate(inference_loader), total=len(inference_loader)):
        if n < done:
            continue

        torch.save(n, log_path)
        video_ids, segments = zip(*targets)
        labels = [class_labels_map[video_ids[i].split("_")[1].lower()] for i in range(len(video_ids))]
        with torch.inference_mode():
            outputs = model(inputs).cpu().numpy()
            preds = outputs.argmax(1)
        if args.use_approx:
            heatmaps = osm.run_videos(inputs, labels)
        else:
            with torch.inference_mode():
                heatmaps = osm.run_videos(inputs, labels)
        
        save_dir = os.path.join(root, video_ids[0])
        os.makedirs(save_dir, exist_ok=True)
        for i in range(len(heatmaps)):
            np.save(os.path.join(save_dir, "{}_label{}_pred{}".format(i, labels[i], preds[i])), 
                heatmaps[i])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt_path", type=str, 
        default="/workspace/data/r3d_models/finetuning/ucf101/r3d50_K_fc/opts.json")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, nargs="+", default=[0, 1])

    parser.add_argument("--map_type", type=str, default="save_results_dir")
    parser.add_argument("--temporal_crops", type=int, nargs="+", default=[8])
    parser.add_argument("--spatial_crops", type=int, nargs="+", default=[32])
    parser.add_argument("--spatial_stride", type=int, default=8)
    parser.add_argument("--temporal_stride", type=int, default=2)
    parser.add_argument("--batchsize", type=int, default=400)
    parser.add_argument("--flow_method", type=str, default="farneback")
    parser.add_argument("--N_stack_mask", type=int, default=1)
    parser.add_argument("--use_approx", action="store_true")

    args = parser.parse_args()
    make(args)