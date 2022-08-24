import shutil

if __name__ == "__main__":
    ###
    target_file = "resnet3d/models/resnet.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if i == 42:
            lines[i] = "        self.relu2 = nn.ReLU(inplace=True)\n"
        if i == 57:
            lines[i] = "        out = self.relu2(out)\n"
        if i == 77:
            lines[i] = "        self.relu2 = nn.ReLU(inplace=True); self.relu3 = nn.ReLU(inplace=True)\n"
        if i == 87:
            lines[i] = "        out = self.relu2(out)\n"
        if i == 96:
            lines[i] = "        out = self.relu3(out)\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/models/resnext.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from utils import partialclass\n":
            lines[i] = "from .utils import partialclass\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/main.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from opts import parse_opts\n":
            lines[i] = "from .opts import parse_opts\n"
        if line == "from model import (generate_model, load_pretrained_model, make_data_parallel,\n":
            lines[i] = "from .model import (generate_model, load_pretrained_model, make_data_parallel,\n"
        if line == "from mean import get_mean_std\n":
            lines[i] = "from .mean import get_mean_std\n"
        if line == "from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,\n":
            lines[i] = "from .spatial_transforms import (Compose, Normalize, Resize, CenterCrop,\n"
        if line == "from temporal_transforms import (LoopPadding, TemporalRandomCrop,\n":
            lines[i] = "from .temporal_transforms import (LoopPadding, TemporalRandomCrop,\n"
        if line == "from temporal_transforms import Compose as TemporalCompose\n":
            lines[i] = "from .temporal_transforms import Compose as TemporalCompose\n"
        if line == "from dataset import get_training_data, get_validation_data, get_inference_data\n":
            lines[i] = "from .dataset import get_training_data, get_validation_data, get_inference_data\n"
        if line == "from utils import Logger, worker_init_fn, get_lr\n":
            lines[i] = "from .utils import Logger, worker_init_fn, get_lr\n"
        if line == "from training import train_epoch\n":
            lines[i] = "from .training import train_epoch\n"
        if line == "from validation import val_epoch\n":
            lines[i] = "from .validation import val_epoch\n"
        if line == "import inference\n":
            lines[i] = "from .inference import inference\n"
        if line == "inference.inference(inference_loader, model, inference_result_path,\n":
            lines[i] = "inference(inference_loader, model, inference_result_path,\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/model.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet\n":
            lines[
                i
            ] = "from .models import resnet, resnet2p1d, pre_act_resnet, wide_resnet, resnext, densenet\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/dataset.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from datasets.videodataset import VideoDataset\n":
            lines[i] = "from .datasets.videodataset import VideoDataset\n"
        if line == "from datasets.videodataset_multiclips import (VideoDatasetMultiClips,\n":
            lines[i] = "from .datasets.videodataset_multiclips import (VideoDatasetMultiClips,\n"
        if line == "from datasets.activitynet import ActivityNet\n":
            lines[i] = "from .datasets.activitynet import ActivityNet\n"
        if line == "from datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5\n":
            lines[i] = "from .datasets.loader import VideoLoader, VideoLoaderHDF5, VideoLoaderFlowHDF5\n"
    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/training.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from utils import AverageMeter, calculate_accuracy\n":
            lines[i] = "from .utils import AverageMeter, calculate_accuracy\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/validation.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from utils import AverageMeter, calculate_accuracy\n":
            lines[i] = "from .utils import AverageMeter, calculate_accuracy\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    ###
    target_file = "resnet3d/inference.py"
    out = open("temp.py", "w")
    f = open(target_file, "r")
    lines = f.readlines()

    for i, line in enumerate(lines):
        if line == "from utils import AverageMeter\n":
            lines[i] = "from .utils import AverageMeter\n"

    out.writelines(lines)
    out.close()
    f.close()

    shutil.move("temp.py", target_file)

    # ###
    # out = open("temp.py", "w")
    # f = open("orgiresnet/CIFAR_main.py", "r")
    # lines = f.readlines()

    # outlines = ["import torch\n"]
    # append = False
    # for i, line in enumerate(lines):
    #     if "def get_init_batch" in line or append:
    #         append = True
    #         outlines.append(line)

    #     if line == "\n" and append:
    #         append = False

    # out.writelines(outlines)
    # out.close()
    # f.close()

    # shutil.move("temp.py", "orgiresnet/utils.py")
