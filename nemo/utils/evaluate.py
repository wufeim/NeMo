import numpy as np
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.transforms import Translate, Rotate
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


class Transform6DPose():
    def __init__(self, azimuth, elevation, theta, distance, principal, img_size=(320, 448), focal_length=3000, device='cpu'):
        R, T = look_at_view_transform([distance], [elevation], [azimuth], degrees=False, device=device)
        self.R = torch.bmm(R, self.rotation_theta(theta, device_=device))
        self.T = T + self.convert_principal_to_translation(distance, principal, img_size, focal_length).to(device)

    def rotation_theta(self, theta, device_=None):
        # cos -sin  0
        # sin  cos  0
        # 0    0    1
        if type(theta) == float:
            if device_ is None:
                device_ = 'cpu'
            theta = torch.ones((1, 1, 1)).to(device_) * theta
        else:
            if device_ is None:
                device_ = theta.device
            theta = theta.view(-1, 1, 1)

        mul_ = torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]]).view(1, 2, 9).to(device_)
        bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

        # [n, 1, 2]
        cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

        # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
        trans = torch.matmul(cos_sin, mul_) + bia_
        trans = trans.view(-1, 3, 3)

        return trans

    def convert_principal_to_translation(self, distance, principal_, image_size_, focal_=3000):
        principal_ = np.array(principal_, dtype=np.float32)
        d_p = torch.Tensor(principal_).float() - torch.Tensor(image_size_).flip(0) / 2
        return torch.Tensor([[-d_p[0] * distance / focal_, -d_p[1] * distance / focal_, 0]])

    def __call__(self, points):
        T_ = Translate(self.T, device=self.T.device)
        R_ = Rotate(self.R, device=self.R.device)
        transforms = R_.compose(T_)
        return transforms.transform_points(points)


def add3d(gt, pred, kps):
    if isinstance(kps, np.ndarray):
        kps = torch.from_numpy(kps).float()
    gt_trans = Transform6DPose(gt['azimuth'], gt['elevation'], float(gt['theta']), gt['distance'], gt['principal'])
    pred_trans = Transform6DPose(pred['azimuth'], pred['elevation'], float(pred['theta']), pred['distance'], pred['principal'])
    pt1 = gt_trans(kps)
    pt2 = pred_trans(kps)
    return np.mean(np.sqrt(np.sum((pt1.detach().cpu().numpy() - pt2.detach().cpu().numpy())**2, axis=1)))
