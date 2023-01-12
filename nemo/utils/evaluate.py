import numpy as np
from scipy.linalg import logm
import torch


def pose_error(gt, pred):
    from nemo.utils import cal_rotation_matrix

    if pred is None:
        return np.pi
    azimuth_gt, elevation_gt, theta_gt = (
        float(gt["azimuth"]),
        float(gt["elevation"]),
        float(gt["theta"]),
    )
    azimuth_pred, elevation_pred, theta_pred = (
        float(pred["azimuth"]),
        float(pred["elevation"]),
        float(pred["theta"]),
    )
    anno_matrix = cal_rotation_matrix(theta_gt, elevation_gt, azimuth_gt, 5.0)
    pred_matrix = cal_rotation_matrix(theta_pred, elevation_pred, azimuth_pred, 5.0)
    if (
        np.any(np.isnan(anno_matrix))
        or np.any(np.isnan(pred_matrix))
        or np.any(np.isinf(anno_matrix))
        or np.any(np.isinf(pred_matrix))
    ):
        error_ = np.pi
    else:
        error_ = (
            (logm(np.dot(np.transpose(pred_matrix), anno_matrix)) ** 2).sum()
        ) ** 0.5 / (2.0 ** 0.5)
    return error_


def iou(mask1, mask2):
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.detach().cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.detach().cpu().numpy()

    intersection = np.sum(mask1 * mask2)
    union = np.sum(mask1) + np.sum(mask2) - intersection
    if union == 0:
        return 1.0
    return intersection / union
