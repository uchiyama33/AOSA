import cv2
import numpy as np
import torch
from PIL import Image
from scipy.stats import gamma as pmodel
from torch.nn import functional as F

from utils.sensitivity_analysis import Base


class ApproxBase(Base):
    def __init__(
        self,
        n_window=1,
        n_split=0,
        approx_type="simple",
        conditional=False,
        adjust_method=None,
        *args,
        **kargs,
    ):
        """[summary]
        Args:
            approx_type (str, optional): not used. Defaults to "simple".
            conditional (bool, optional): apply conditional sampling or not. Defaults to False.
            adjust_method (str, optional): decide adjustment method. available values [simple, ip, None]. Defaults to None.
        """
        self.approx_type = approx_type
        super().__init__(*args, **kargs)
        self.conditional = conditional
        self.model_type = None
        self.n_split = n_split

        if adjust_method == "ip":
            self.adjust_method = self.ip_base_adjustment
        elif adjust_method is not None:
            self.adjust_method = self.simple_adjustment
        else:
            self.adjust_method = None

        if conditional:
            filter = torch.ones((1, 1, n_window, n_window)) / (n_window * n_window)
            self.filter = filter.to(self.device)
        else:
            self.filter = None

        if self.n_split > 0:
            tmax, hmax, wmax = self.heat_size
            nt, nh, nw = np.ceil(np.asarray(self.heat_size) / n_split).astype(np.uint8)
            if nt <= 1 or nh <= 1 or nw <= 1:
                self.n_split = 0
                print("#split is too large. ignore")
            else:  # TODO below
                self.inv_masks = [
                    [
                        m.reshape(self.heat_size + list(m.shape[1:]))[
                            i * nh : min((i + 1) * nh, hmax),
                            j * nw : min((j + 1) * nw, wmax),
                            ...,
                        ].reshape([-1] + list(m.shape[1:]))
                        for i in range(n_split)
                        for j in range(n_split)
                        if i * nh < self.heat_size[0] and j * nw < self.heat_size[1]
                    ]
                    for m in self.inv_masks
                ]
                self.mean_m = [[m_grp.mean(dim=0) for m_grp in m] for m in self.inv_masks]

                self.ids = [
                    np.hstack(
                        [
                            k + np.asarray(range(j * nw, min((j + 1) * nw, wmax)))
                            for k in wmax * np.asarray(range(i * nh, min((i + 1) * nh, hmax)))
                        ]
                    )
                    for i in range(n_split)
                    for j in range(n_split)
                    if i * nh < self.heat_size[0] and j * nw < self.heat_size[1]
                ]

    def _post_process(self, org_prob, _probs):
        return self._normalize(
            _probs,
            org_prob,
        )

    def _approximate(self, fa, fa_grad, x, v, m):
        with torch.inference_mode():
            # m = m[0]
            # m = 1 - _m
            N = m.shape[0]
            BS = self.batchsize
            sum_idx = list(range(1, fa_grad.dim()))

            with torch.inference_mode():
                diff = v - x
                grad_diff = fa_grad * diff
                f = [
                    fa + (grad_diff * m[i : min(i + BS, N), ...].to(self.device)).sum(sum_idx)
                    for i in range(0, N, BS)
                ]

        return torch.cat(f)  # torch.hstack(f)

    def _get_grad(self, org_video, target_class):
        if isinstance(org_video, Image.Image):
            x, trans_video, fa = self._forward(org_video, target_class=target_class, requires_grad=True)
        else:
            x = org_video.clone()
            trans_video = []
            for i in range(self.video_size[2]):
                img = org_video.squeeze().transpose(0, 1)[i]
                trans_video.append(self.unnormalize(img))
            x, fa = self._forward(x.to(self.device), target_class=target_class, requires_grad=True)

        fa.backward()
        fa_grad = x.grad

        return x, fa, fa_grad, trans_video

    def _adjustment(self, x, v, point_id, target_ids, target_class):
        tx = (x * self.masks[point_id, ...].to(x)).squeeze().detach()
        x, fa, fa_grad, _ = self._get_grad(tx, target_class)

        _m = self.inv_masks[target_ids, ...]

        return self._approximate(fa, fa_grad, x, v, _m)

    def _lower_upper(self, data):
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return lower, upper

    def simple_adjustment(self, f, fa, x, v, target_class):
        _p = self.adjust_method
        diff = (f - fa).squeeze().detach().cpu().numpy()

        # idx = np.where(diff > pmodel.ppf(_p, *pmodel.fit(diff[diff > 0])))[0]
        # _idx = np.argsort(diff)
        # idx = _idx[-int(len(_idx) * _p) :]

        lower, upper = self._lower_upper(diff)

        idx = np.where(diff > upper)[0]
        if len(idx) > 0:
            f[idx] = self._adjustment(x, v, diff.argmax(), idx, target_class)

        # idx = np.where(diff < -pmodel.ppf(_p, *pmodel.fit(-diff[diff < 0])))[0]
        # idx = _idx[: int(len(_idx) * _p)]

        idx = np.where(diff < lower)[0]
        if len(idx) > 0:
            f[idx] = self._adjustment(x, v, diff.argmin(), idx, target_class)
        return f

    def ip_base_adjustment(self, f, fa, x, v, t, s, target_class):
        def _connected_components(diff, _idx):
            m = np.zeros(diff.shape)
            m[_idx] = 1
            kernel = np.ones(
                (
                    int(self.heat_size[0] * 0.2 + 1),
                    int(self.heat_size[1] * 0.2 + 1),
                    int(self.heat_size[2] * 0.2 + 1),
                ),
                np.uint8,
            )
            m = m.reshape(self.heat_size).astype(np.uint8)
            # マスクの数がself.heat_sizeの積と同じと想定されている
            mdilate = cv2.dilate(m, kernel)

            _, labels = cv2.connectedComponents(mdilate)
            ids = [np.where((m * labels).astype(np.uint8) == _i) for _i in range(1, labels.max() + 1)]
            ids = [np.sort(i[1] + (i[0] * self.heat_size[1])) for i in ids]
            return ids

        diff = (f - fa).squeeze().detach().cpu().numpy()
        lower, upper = self._lower_upper(diff)

        # idx = np.where(diff < -pmodel.ppf(_p, *pmodel.fit(-diff[diff < 0])))

        # _idx = np.argsort(diff)
        # idx = _idx[: int(len(_idx) * _p)]

        idx = np.where(diff < lower)[0]
        ids = _connected_components(diff, idx)
        for _id in ids:
            base_id = _id[diff[_id].argmin()]
            f[_id] = self._adjustment(x, v, t, s, base_id, _id, target_class)

        # idx = np.where(diff > pmodel.ppf(_p, *pmodel.fit(diff[diff > 0])))

        # idx = _idx[-int(len(_idx) * _p) :]
        idx = np.where(diff > upper)[0]
        ids = _connected_components(diff, idx)
        for _id in ids:
            base_id = _id[diff[_id].argmax()]
            f[_id] = self._adjustment(x, v, t, s, base_id, _id, target_class)

        return f

    def _run(self, org_video, target_class):
        if self.n_split == 0:
            x, fa, fa_grad, trans_video = self._get_grad(org_video, target_class)

            self.masks = self._gen_masks(
                trans_video,
                self.spatial_crop_size,
                self.temporal_crop_size,
                self.video_size,
                self.spatial_stride,
                self.temporal_stride,
            )

            self.N = self.masks.shape[0]
            self.inv_masks = 1 - self.masks

            if self.conditional:
                with torch.inference_mode():
                    v = (
                        F.conv2d(
                            x.squeeze().reshape(1, -1, x.shape[3], x.shape[4]).transpose(0, 1),
                            self.filter,
                            padding="same",
                        )
                        .reshape(self.video_size)
                        .squeeze()
                    )
                    # v = F.conv3d(x.squeeze().unsqueeze(1), self.filter, padding="same").squeeze()
            else:
                v = self.rep_vals

            f = self._approximate(fa, fa_grad, x, v, self.inv_masks)

            if self.adjust_method is not None:
                f = self.adjust_method(f, fa, x, v, target_class)

            results = self._post_process(fa, f)
            return results

        hmax, wmax = self.heat_size
        f = torch.zeros(hmax * wmax).to(self.device)
        if isinstance(org_video, Image.Image):
            x, _, org_fa = self._forward(org_video, target_class=target_class, requires_grad=True)
        else:
            x = org_video.clone()
            x, org_fa = self._forward(x, target_class=target_class, requires_grad=True)

        if self.conditional:
            with torch.inference_mode():
                v = F.conv2d(x.squeeze().unsqueeze(1), self.filter, padding="same").squeeze()
        else:
            v = self.rep_vals

        for cid in range(len(self.inv_masks)):
            m = self.inv_masks[cid]
            _mean_m = self.mean_m[cid]
            for m_grp, idx, mean_m in zip(m, self.ids, _mean_m):
                # average image of masked images
                # taylor expansion at tx
                tx = (x * (1 - mean_m) + v * mean_m).squeeze().detach()
                _, fa, fa_grad, trans_video = self._get_grad(tx, target_class)
                f[idx] = self._approximate(fa, fa_grad, x, v, m_grp)

            # f = (f[:, None, None] * aosm.inv_masks[0][:, 0, ...]).sum(0) / aosm.inv_masks[0
            # ][:, 0, ...].sum(0)
            # map = [normalize_heatmap(f.detach().cpu().numpy(), fa.detach().cpu().numpy())]
            if self.adjust_method is not None:
                f = self.adjust_method(f, org_fa, x, v, cid, target_class)

            results[cid] = self._post_process(org_fa, f)
        return results


class ApproxOcclusionSensitivityMap3D(ApproxBase):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

    def run(self, org_video, target_class):
        return self._run(org_video, target_class)
