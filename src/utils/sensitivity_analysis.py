import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn.functional import softmax
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Normalize, ToPILImage
from copy import deepcopy
from time import time

from utils.subspace_model import SubspaceModel
from utils.utils import normalize_heatmap
import org3dresnet
from mmflow.apis import init_model, inference_model


class Base:
    def __init__(
        self,
        video_size,
        device,
        spatial_crop_sizes,
        temporal_crop_sizes,
        spatial_stride=2,
        temporal_stride=2,
        crop_type="",
        transform=None,
        batchsize=64,
        net=None,
        use_softmax=True,
        N_stack_mask=1,
        N_mask_set=1,
        gen_mask="flow_one",
        flow_method="farneback",
        save_inputs_path=None,
        normalize_each_frame=False,
        stack_method="flow_vec_corr",
        delete_point=True,
        delete_outside=True,
        consider_letter_box=True,
        median_filter=False,
    ):
        self.spatial_crop_sizes = spatial_crop_sizes
        self.temporal_crop_sizes = temporal_crop_sizes
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.crop_type = crop_type
        self.device = device
        self.batchsize = batchsize
        self.use_softmax = use_softmax
        self.video_size = video_size
        self.N_stack_mask = N_stack_mask
        self.N_mask_set = N_mask_set
        self.gen_mask = gen_mask
        self.flow_method = flow_method
        self.save_inputs_path = save_inputs_path
        self.normalize_each_frame = normalize_each_frame
        self.stack_method = stack_method
        self.delete_point = delete_point
        self.delete_outside = delete_outside
        self.consider_letter_box = consider_letter_box
        self.median_filter = median_filter

        assert type(N_mask_set) == int

        self.net = net
        self.net = self.net.eval()

        self.transform = transform
        try:
            # make unnormalization
            _transforms = self.transform.transforms
            idx = [type(i) for i in _transforms].index(
                org3dresnet.spatial_transforms.Normalize)
            self.normalize = _transforms[idx]
            mean = torch.tensor(self.normalize.mean)
            std = torch.tensor(self.normalize.std)

            self.unnormalize = transforms.Compose(
                [
                    Normalize((-mean / std).tolist(), (1 / std).tolist()),
                    ToPILImage(),
                ]
            )
        except:
            self.unnormalize = transforms.Compose([ToPILImage()])

        self.rep_vals = self._gen_replacevals(video_size)

        self.heat_size = [
            len(range(0, video_size[2], temporal_stride)),
            len(range(0, video_size[3], spatial_stride)),
            len(range(0, video_size[4], spatial_stride)),
        ]

        self._init_mmflow_model()

    def _init_mmflow_model(self):
        if self.flow_method == "gma":
            config_file = "/workspace/data/mmflow/gma_8x2_120k_mixed_368x768.py"
            checkpoint_file = "/workspace/data/mmflow/gma_8x2_120k_mixed_368x768.pth"
            self.flow_model = init_model(
                config_file, checkpoint_file, device='cuda:0')

        if self.flow_method == "liteflownet2":
            config_file = "/workspace/data/mmflow/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.py"
            checkpoint_file = "/workspace/data/mmflow/liteflownet2_ft_4x1_600k_sintel_kitti_320x768.pth"
            self.flow_model = init_model(
                config_file, checkpoint_file, device='cuda:0')

        if self.flow_method == "pwcnet":
            config_file = "/workspace/data/mmflow/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.py"
            checkpoint_file = "/workspace/data/mmflow/pwcnet_plus_8x1_750k_sintel_kitti2015_hd1k_320x768.pth"
            self.flow_model = init_model(
                config_file, checkpoint_file, device='cuda:0')

    def _gen_masks(self, video, spatial_crop_size, temporal_crop_size, video_size, spatial_stride, temporal_stride):
        m = []
        fxfy = []
        spatial_h_csize = spatial_crop_size // 2
        spatial_offset = spatial_crop_size % 2
        temporal_h_csize = temporal_crop_size // 2
        temporal_offset = temporal_crop_size % 2

        if self.consider_letter_box:
            self.has_letter_box = self._check_letter_box(video)
        else:
            self.has_letter_box = False

        if self.gen_mask == "simple":
            for k in range(0, video_size[2], temporal_stride):
                for i in range(0, video_size[3], spatial_stride):
                    for j in range(0, video_size[4], spatial_stride):
                        _m = torch.ones(video_size)
                        top = max(0, i - spatial_h_csize)
                        bottom = min(video_size[3], i +
                                     spatial_h_csize + spatial_offset)

                        left = max(0, j - spatial_h_csize)
                        right = min(video_size[4], j +
                                    spatial_h_csize + spatial_offset)

                        front = max(0, k - temporal_h_csize)
                        back = min(video_size[2], k +
                                   temporal_h_csize + temporal_offset)
                        _m[..., front:back, top:bottom, left:right] = 0
                        m.append(_m)

            m = torch.stack(m).unsqueeze(1)

        elif self.gen_mask == "flow":
            flow_list = self._calc_optical_flow(video)  # 0.72s

            T = self.video_size[2]
            H = self.video_size[3]
            W = self.video_size[4]

            temporal_step = np.arange(0, T, temporal_stride)
            start_y, start_x = np.mgrid[spatial_stride/2:H:spatial_stride,
                                        spatial_stride/2:W:spatial_stride].reshape(2, -1).astype(int)
            if self.has_letter_box:  # FIX
                start_x = start_x[(start_y >= 12) & (start_y <= 100)]
                start_y = start_y[(start_y >= 12) & (start_y <= 100)]
            m = []

            use_start_tracking_points = np.full(
                (len(temporal_step), len(start_y)), True)
            # debug
            self.flow_x = np.zeros((len(temporal_step), len(start_y), T))
            self.flow_y = np.zeros((len(temporal_step), len(start_y), T))
            self.flow_z = np.zeros((len(temporal_step), len(start_y), T))

            for k, t_step in enumerate(temporal_step):
                _fxfy = []
                y, x = start_y, start_x
                y = y[use_start_tracking_points[k]]
                x = x[use_start_tracking_points[k]]
                keep_tracking_points = np.full(len(y), True)
                _m = torch.ones(tuple([len(y)]) + tuple(video_size))

                # start = max(0, t_step - temporal_h_csize)
                # end = min(T, t_step + temporal_h_csize + temporal_offset)
                start = t_step
                end = min(T, t_step + temporal_crop_size)
                for t in range(0, end):
                    if t < start:
                        self.flow_x[k][grid_cnt][t] = 0
                        self.flow_y[k][grid_cnt][t] = 0
                        self.flow_z[k][grid_cnt][t] = 0
                        _fxfy.append(np.zeros_like(flow_list[0][y, x]))
                    else:
                        grid_cnt = 0
                        for grid_cnt, (i, j) in enumerate(zip(y, x)):
                            if keep_tracking_points[grid_cnt]:
                                # debug
                                self.flow_x[k][grid_cnt][t] = j
                                self.flow_y[k][grid_cnt][t] = i
                                self.flow_z[k][grid_cnt][t] = t

                                top = max(0, i - spatial_h_csize)
                                bottom = min(video_size[3], i +
                                             spatial_h_csize + spatial_offset)

                                left = max(0, j - spatial_h_csize)
                                right = min(video_size[4], j +
                                            spatial_h_csize + spatial_offset)

                                _m[grid_cnt, :, :, t, top:bottom, left:right] = 0
                            else:
                                self.flow_x[k][grid_cnt][t] = 0
                                self.flow_y[k][grid_cnt][t] = 0
                                self.flow_z[k][grid_cnt][t] = 0

                        if t < T-1:
                            pre_y = y
                            pre_x = x
                            fx, fy = flow_list[t][y, x].T
                            y = np.int32(y + fy).clip(0, video_size[3]-1)
                            x = np.int32(x + fx).clip(0, video_size[4]-1)
                            _fxfy.append(np.array([fx, fy]).T)

                        if self.delete_outside:
                            keep_tracking_points = self._delete_outside_screen(
                                pre_y, pre_x, fy, fx, keep_tracking_points)

                        if self.delete_point and t in temporal_step[k+1:]:
                            for grid_cnt, (i, j) in enumerate(zip(y, x)):
                                hit_y = np.abs(start_y-i) <= spatial_h_csize
                                hit_x = np.abs(start_x-j) <= spatial_h_csize
                                hit_yx = hit_y & hit_x
                                use_start_tracking_points[temporal_step == t, hit_yx] = False

                m.append(_m)
                _fxfy = np.array(_fxfy).transpose(1, 0, 2)
                fxfy.append(_fxfy)

            fxfy = np.concatenate(fxfy).transpose(1, 2, 0)
            m = torch.cat(m).unsqueeze(1)

        elif self.gen_mask == "flow_one":
            flow_list = self._calc_optical_flow(video)

            T = self.video_size[2]
            H = self.video_size[3]
            W = self.video_size[4]
            y, x = np.mgrid[spatial_stride/2:H:spatial_stride,
                            spatial_stride/2:W:spatial_stride].reshape(2, -1).astype(int)
            if self.has_letter_box:  # FIX
                x = x[(y >= 12) & (y <= 100)]
                y = y[(y >= 12) & (y <= 100)]
            m = torch.ones(tuple([len(y)]) + tuple(video_size))

            # debug
            self.flow_x = [[None] * T for i in range(m.shape[0])]
            self.flow_y = [[None] * T for i in range(m.shape[0])]
            self.flow_z = [[None] * T for i in range(m.shape[0])]

            keep_tracking_points = np.full(len(y), True)

            for t in range(0, T):
                for grid_cnt, (i, j) in enumerate(zip(y, x)):
                    if keep_tracking_points[grid_cnt]:
                        # debug
                        self.flow_x[grid_cnt][t] = j
                        self.flow_y[grid_cnt][t] = i
                        self.flow_z[grid_cnt][t] = t

                        top = max(0, i - spatial_h_csize)
                        bottom = min(video_size[3], i +
                                     spatial_h_csize + spatial_offset)

                        left = max(0, j - spatial_h_csize)
                        right = min(video_size[4], j +
                                    spatial_h_csize + spatial_offset)

                        m[grid_cnt, :, :, t, top:bottom, left:right] = 0

                    else:
                        self.flow_x[grid_cnt][t] = 0
                        self.flow_y[grid_cnt][t] = 0
                        self.flow_z[grid_cnt][t] = 0

                if t < T-1:
                    pre_y = y
                    pre_x = x
                    fx, fy = flow_list[t][y, x].T
                    y = np.round(y + fy).astype(np.int16).clip(0,
                                                               video_size[3]-1)
                    x = np.round(x + fx).astype(np.int16).clip(0,
                                                               video_size[4]-1)
                    fxfy.append(np.array([fx, fy]))

                    if self.delete_outside:
                        keep_tracking_points = self._delete_outside_screen(
                            pre_y, pre_x, fy, fx, keep_tracking_points)

            m = m.unsqueeze(1)
            fxfy = np.array(fxfy)

        else:
            assert False, "gen_mask"

        if self.N_stack_mask != 1:
            m = self._stack_mask_method(m, fxfy)    # 0.27s

        return m

    def _stack_mask_method(self, m, fxfy):
        if self.stack_method == "random":
            m = self._random_stack_mask(m)
        elif self.stack_method == "near_norm" and self.gen_mask == "flow_one":
            m = self._norm_stack_mask(m, fxfy)
        elif self.stack_method == "flow_norm_corr":
            m = self._flow_norm_corr_stack_mask(m, fxfy)
        elif self.stack_method == "flow_vec_corr":
            m = self._flow_vec_corr_stack_mask(m, fxfy)
        else:
            assert False, "stack_method"

        return m

    def _check_letter_box(self, video):
        max_value = 500000
        sum_value = 0
        for v in video:
            v = np.array(v)
            sum_value = sum_value + v[:3, :, :].sum()
            sum_value = sum_value + v[-3:, :, :].sum()
        if sum_value < max_value:
            return True
        else:
            return False

    def _delete_outside_screen(self, pre_y, pre_x, fy, fx, keep_tracking_points):
        if self.has_letter_box:
            out_y = ((pre_y + fy) < 16) | ((pre_y + fy) > 96)
        else:
            out_y = ((pre_y + fy) < 0) | ((pre_y + fy) > 111)
        out_x = ((pre_x + fx) < 0) | ((pre_x + fx) > 111)
        keep_tracking_points[out_y | out_x] = False
        return keep_tracking_points

    def _remove_all_one_mask(self, m):
        _m = []
        for i in range(len(m)):
            if not torch.allclose(m[i], torch.ones(self.video_size)):
                _m.append(m[i])
        return torch.stack(_m)

    def get_flow_xyz(self):
        return self.flow_x, self.flow_y, self.flow_z

    def _calc_optical_flow(self, video, ):
        if self.flow_method == "farneback":
            flow_list = []
            prvs = cv2.cvtColor(np.array(video[0]), cv2.COLOR_BGR2GRAY)
            for i in range(1, self.video_size[2]):
                next = cv2.cvtColor(np.array(video[i]), cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                flow_list.append(flow)
                prvs = next

        # elif self.flow_method == "pyflow":
        #     alpha = 0.012
        #     ratio = 0.5
        #     minWidth = 20
        #     nOuterFPIterations = 7
        #     nInnerFPIterations = 1
        #     nSORIterations = 30
        #     # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
        #     colType = 1
        #     flow_list = []
        #     prvs = cv2.cvtColor(np.array(video[0]), cv2.COLOR_BGR2GRAY)[
        #         :, :, None].astype(float) / 255.
        #     for i in range(1, self.video_size[2]):
        #         next = cv2.cvtColor(np.array(video[i]), cv2.COLOR_BGR2GRAY)[
        #             :, :, None].astype(float) / 255.
        #         u, v, _ = pyflow.coarse2fine_flow(
        #             prvs,
        #             next,
        #             alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        #             nSORIterations, colType)
        #         flow = np.concatenate((u[..., None], v[..., None]), axis=2)
        #         flow_list.append(flow)
        #         prvs = next

        elif self.flow_method in ["gma", "liteflownet2", "pwcnet"]:
            flow_list = []
            prvs = np.array(video[0])
            for i in range(1, self.video_size[2]):
                next = np.array(video[i])
                with torch.inference_mode():
                    flow = inference_model(self.flow_model, prvs, next)
                flow_list.append(flow)
                prvs = next

        else:
            assert False, "unknown flow method"

        if self.median_filter:
            for i in range(self.video_size[2] - 1):
                flow_list[i] = cv2.medianBlur(flow_list[i], 3)

        return flow_list

    def _norm_stack_mask(self, mask, fxfy):
        fxfy = np.array(fxfy)
        flow_norm = np.linalg.norm(fxfy, axis=1).sum(0)

        stacked_mask = []
        for i in range(len(flow_norm)):
            near = np.abs(flow_norm - flow_norm[i])
            near[i] = 1e5

            idx_list = near.argsort()[:self.N_stack_mask]

            _m = torch.cat(
                [mask[[i]], mask[idx_list]])
            stacked_mask.append(torch.prod(_m, 0))

        return torch.stack(stacked_mask)

    def _flow_vec_corr_stack_mask(self, mask, fxfy):
        flow_vec = fxfy.reshape(-1, fxfy.shape[2]).T
        norm = np.linalg.norm(flow_vec, axis=1)
        R = flow_vec @ flow_vec.T

        mask = mask.to(self.device)
        stacked_mask = []
        for i in range(len(flow_vec)):
            sims = deepcopy(R[i]) / (norm + 1e-10) / (norm[i] + 1e-10)
            sims[i] = 0

            idx_list = sims.argsort()[-self.N_stack_mask:]

            _m = torch.cat(
                [mask[[i]], mask[idx_list]])
            stacked_mask.append(torch.prod(_m, 0))

        return torch.stack(stacked_mask).cpu()

    def _flow_norm_corr_stack_mask(self, mask, fxfy):
        norm_vec = np.linalg.norm(fxfy, axis=1).T
        R = norm_vec @ norm_vec.T
        norm_vec_norm = np.linalg.norm(norm_vec, axis=1)

        mask = mask.to(self.device)
        stacked_mask = []
        for i in range(len(norm_vec)):
            sims = deepcopy(R[i]) / (norm_vec_norm + 1e-10) / \
                (norm_vec_norm[i] + 1e-10)
            sims[i] = 0

            idx_list = sims.argsort()[-self.N_stack_mask:]

            _m = torch.cat(
                [mask[[i]], mask[idx_list]])
            stacked_mask.append(torch.prod(_m, 0))

        return torch.stack(stacked_mask).cpu()

    def _random_stack_mask(self, mask):
        N = mask.shape[0]
        rand_idx = []
        for i in range(self.N_mask_set):
            rand_idx.append(torch.randperm(N))
        rand_idx = torch.cat(rand_idx)
        mask = mask[rand_idx]
        stacked_mask = []
        for i in range(0, N*self.N_mask_set, self.N_stack_mask):
            _m = mask[i:i+self.N_stack_mask]
            _m = torch.prod(_m, 0)
            stacked_mask.append(_m)

        return torch.stack(stacked_mask)

    def _gen_replacevals(self, video_size):
        return torch.zeros(video_size, device=self.device)

    def _gen_input(self, org_tensor, spatial_crop_id, temporal_crop_id):
        m = self.masks[temporal_crop_id][spatial_crop_id]
        m = m[:, 0, 0, ...]
        occ_videos = org_tensor * m
        return occ_videos

    def _save_unnorm_inputs(self, inputs):
        _inputs = inputs.clone()
        unnorm_inputs = []
        for _input in tqdm(_inputs, leave=False, desc="save_unnorm_inputs"):
            _input = _input.transpose(0, 1)
            _unnorm_input = []
            for _img in _input:
                _unnorm_input.append(self.unnormalize(_img))
            unnorm_inputs.append(_unnorm_input)

        os.makedirs(os.path.dirname(self.save_inputs_path), exist_ok=True)
        torch.save(unnorm_inputs, self.save_inputs_path)

    def _normalize(
        self,
        X,
        trans_video,
        org_val,
        spatial_crop_id,
        temporal_crop_id,
        model_type=None,
        approx=False,
    ):
        X = X - org_val
        X = X.cpu()
        _mask = self.masks[temporal_crop_id][spatial_crop_id]
        _map = X @ (_mask.view(self.N, -1) /
                    _mask.reshape(_mask.shape[0], -1).mean(1)[:, np.newaxis])
        map = -_map.cpu().view(self.video_size[1:]).mean(0) / self.N

        map = (map - map.min()) / (map - map.min()).max()
        map = map * (X.max() - X.min())
        map = map + X.min()

        if self.normalize_each_frame:
            for i in range(len(map)):
                map[i] = normalize_heatmap(map[i], 0, model_type)
        else:
            map = normalize_heatmap(map, 0, model_type)

        return map.numpy()

    def _forward(self, x, requires_grad=False, target_class=None):
        if type(x) == Image.Image:
            # forward pil image
            tensor = self.transform(x).unsqueeze(0)
            tensor.requires_grad = requires_grad
            prob = self.net(tensor)

            if self.use_softmax:
                prob = softmax(prob, dim=0)

            if target_class is not None:
                prob = prob.squeeze()[target_class]

            _img = self.unnormalize(tensor.squeeze())
            return tensor, _img, prob.squeeze()

        else:
            if x.dim() == 4:
                x = x.unsqueeze(0)
            # forward inputs
            x.requires_grad = requires_grad
            bs = self.batchsize
            N = x.shape[0]
            _probs = [
                self.net(x[j: min([j + bs, N]), ...]) for j in range(0, N, bs)
            ]
            _probs = torch.vstack(_probs)
            if self.use_softmax:
                _probs = softmax(_probs, dim=1)

            if requires_grad:
                if target_class is not None:
                    return x, _probs[:, target_class].squeeze()
                else:
                    return x, _probs.squeeze()

            if target_class is not None:
                return _probs[:, target_class].squeeze()
            else:
                return _probs.squeeze()

    def _post_process(self):
        raise NotImplementedError

    def _run(
        self,
        org_tensor,
        org_prob,
        target_class,
        trans_video,
        get_feat=False,
    ):
        self.masks = [[
            self._gen_masks(trans_video, s_cs, t_cs, self.video_size,
                            self.spatial_stride, self.temporal_stride)
            for s_cs in self.spatial_crop_sizes] for t_cs in self.temporal_crop_sizes]
        self.N = self.masks[0][0].shape[0]

        results = [[[] for _ in self.spatial_crop_sizes]
                   for _ in self.temporal_crop_sizes]
        for i in range(len(self.temporal_crop_sizes)):
            for j in range(len(self.spatial_crop_sizes)):
                inputs = self._gen_input(org_tensor, j, i)
                if self.save_inputs_path:
                    self._save_unnorm_inputs(inputs)

                if get_feat:
                    _probs = self._forward(inputs)
                else:
                    _probs = self._forward(inputs, target_class=target_class)

                results[i][j] = self._post_process(
                    org_prob, _probs, target_class, trans_video, j, i
                )
        return results

    def run(self):
        raise NotImplementedError

    def run_videos(self, org_videos, target_classses):
        results = []
        for org_video, t_class in tqdm(zip(org_videos, target_classses),
                                       total=len(target_classses), leave=False):
            results.append(self.run(org_video, t_class))
        return results


class OcclusionSensitivityMap3D(Base):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def _post_process(self, org_prob, _probs, target_class, trans_video, s_cs_id, t_cs_id):
        return [
            self._normalize(
                _probs,
                trans_video,
                org_prob,
                s_cs_id,
                t_cs_id
            )
        ]

    def run(self, org_video, target_class):
        with torch.inference_mode():
            if isinstance(org_video, Image.Image):    # 未検査
                org_tensor, trans_video, org_feat = self._forward(
                    org_video, target_class=target_class
                )
            else:
                org_tensor = org_video.clone()
                trans_video = []
                for i in range(self.video_size[2]):
                    img = org_tensor.squeeze().transpose(0, 1)[i]
                    trans_video.append(self.unnormalize(img))
                # trans_video = torch.stack(trans_video)
                org_feat = self._forward(org_video, target_class=target_class)

            results = self._run(org_tensor, org_feat,
                                target_class, trans_video)
        return results
