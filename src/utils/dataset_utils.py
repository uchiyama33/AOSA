import glob
import os
from typing import Iterator, List

import numpy as np
import torch
import torchvision
from PIL import Image
from resnet3d.datasets.activitynet import ActivityNet
from resnet3d.datasets.loader import VideoLoader, VideoLoaderFlowHDF5, VideoLoaderHDF5
from resnet3d.datasets.videodataset import VideoDataset
from resnet3d.datasets.videodataset_multiclips import VideoDatasetMultiClips, collate_fn
from resnet3d.spatial_transforms import (
    CenterCrop,
    Compose,
    Normalize,
    PickFirstChannels,
    Resize,
    ScaleValue,
    ToTensor,
)
from resnet3d.temporal_transforms import Compose as TemporalCompose
from resnet3d.temporal_transforms import SlidingWindow, TemporalSubsampling
from torchvision import get_image_backend, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader

from utils.utils import numericalSort


class ClassBatchSampler(torch.utils.data.sampler.BatchSampler):
    def __init__(self, dataset, target_class, batch_size, shuffle=False):
        self.labels = torch.tensor(dataset.targets)

        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {
            label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set
        }
        if shuffle:
            for l in self.labels_set:
                np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.target_class = target_class
        self.num_samples = len(self.label_to_indices[target_class])
        self.batch_size = batch_size
        self.dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:
        self.count = 0
        # while self.count < self.num_samples:
        for i in range(len(self)):
            indices = []
            indices.extend(
                self.label_to_indices[self.target_class][
                    self.used_label_indices_count[self.target_class] : self.used_label_indices_count[
                        self.target_class
                    ]
                    + self.batch_size
                ]
            )
            self.used_label_indices_count[self.target_class] += self.batch_size

            self.count += self.batch_size

            if self.count >= self.num_samples:
                np.random.shuffle(self.label_to_indices[self.target_class])
                self.used_label_indices_count[self.target_class] = 0
            yield indices

    def __len__(self):
        return self.num_samples // self.batch_size + 1


def make_class_loader(dataset, target_class, batch_size=256, num_workers=6):
    sampler = ClassBatchSampler(dataset, target_class, batch_size)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    return loader


def image_name_formatter(x):
    return f"image_{x:05d}.jpg"


def get_training_data_org(
    video_path,
    annotation_path,
    dataset_name,
    input_type,
    file_type,
    spatial_transform=None,
    temporal_transform=None,
    target_transform=None,
):
    assert dataset_name in ["kinetics", "activitynet", "ucf101", "hmdb51", "mit"]
    assert input_type in ["rgb", "flow"]
    assert file_type in ["jpg", "hdf5"]

    if file_type == "jpg":
        assert input_type == "rgb", "flow input is supported only when input type is hdf5."

        if get_image_backend() == "accimage":
            from resnet3d.datasets.loader import ImageLoaderAccImage

            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = lambda root_path, label, video_id: root_path / label / video_id
    else:
        if input_type == "rgb":
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = lambda root_path, label, video_id: root_path / label / f"{video_id}.hdf5"

    if dataset_name == "activitynet":
        training_data = ActivityNet(
            video_path,
            annotation_path,
            "training",
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
        )
    else:
        training_data = VideoDatasetMultiClips(
            video_path,
            annotation_path,
            "training",
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            video_loader=loader,
            video_path_formatter=video_path_formatter,
        )

    return training_data, collate_fn


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def get_train_loader_org(opt):
    assert opt.train_crop in ["random", "corner", "center"]

    normalize = get_normalize_method(opt.mean, opt.std, opt.no_mean_norm, opt.no_std_norm)

    spatial_transform = [Resize(opt.sample_size)]
    if opt.inference_crop == "center":
        spatial_transform.append(CenterCrop(opt.sample_size))
    spatial_transform.append(ToTensor())
    if opt.input_type == "flow":
        spatial_transform.append(PickFirstChannels(n=2))
    spatial_transform.extend([ScaleValue(opt.value_scale), normalize])
    spatial_transform = Compose(spatial_transform)

    temporal_transform = []
    if opt.sample_t_stride > 1:
        temporal_transform.append(TemporalSubsampling(opt.sample_t_stride))
    temporal_transform.append(SlidingWindow(opt.sample_duration, opt.inference_stride))
    temporal_transform = TemporalCompose(temporal_transform)

    train_data, collate_fn = get_training_data_org(
        opt.video_path,
        opt.annotation_path,
        opt.dataset,
        opt.input_type,
        opt.file_type,
        spatial_transform,
        temporal_transform,
    )
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None

    from resnet3d.datasets.videodataset_multiclips import collate_fn

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=1,
        shuffle=False,
        num_workers=opt.n_threads,
        pin_memory=False,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    return train_loader


def load_jpg_ucf101(l, g, c, n, inference_class_names, transform):
    name = inference_class_names[l]
    dir = os.path.join(
        "/workspace/data/ucf101/jpg", name, "v_{}_g{}_c{}".format(name, str(g).zfill(2), str(c).zfill(2))
    )
    path = sorted(glob.glob(dir + "/*"), key=numericalSort)

    target_path = path[n * 16 : (n + 1) * 16]
    if len(target_path) < 16:
        print("not exist")
        return False

    video = []
    for _p in target_path:
        video.append(transform(Image.open(_p)))

    return torch.stack(video)


def load_jpg_kinetics700(l, v, n, inference_class_names, transform):
    name = inference_class_names[l]
    class_dir = os.path.join("/workspace/data/kinetics-700/val_jpg", name)
    video_jpg_dir = sorted(glob.glob(class_dir + "/*"), key=numericalSort)
    path = sorted(glob.glob(video_jpg_dir[v] + "/*"), key=numericalSort)

    target_path = path[n * 16 : (n + 1) * 16]
    if len(target_path) < 16:
        print("not exist")
        return False

    video = []
    for _p in target_path:
        video.append(transform(Image.open(_p)))

    return torch.stack(video)
