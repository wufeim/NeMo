import copy
import math
import time

import numpy as np
import torch
from pytorch3d.renderer import camera_position_from_spherical_angles
from skimage.feature import peak_local_max

from nemo.utils import call_func_by_name
from nemo.utils import camera_position_to_spherical_angle
from nemo.utils import construct_class_by_name
from nemo.utils import flow_warp
from nemo.utils import load_off


def loss_fg_only(obj_s, clu_s=None, device="cuda"):
    return torch.ones(1, device=device) - torch.mean(obj_s)


def loss_fg_bg(obj_s, clu_s, device="cuda"):
    return torch.ones(1, device=device) - (
        torch.mean(torch.max(obj_s, clu_s)) - torch.mean(clu_s)
    )


def get_corr_pytorch(
    px_samples,
    py_samples,
    kpt_score_map,
    kp_coords,
    kp_vis,
    down_sample_rate,
    hm_h,
    hm_w,
    batch_size=12,
    device="cuda",
):
    all_corr = []

    begin, end = 0, batch_size
    while begin < len(px_samples):
        px_s, py_s = torch.from_numpy(px_samples[begin:end]).to(
            device
        ), torch.from_numpy(py_samples).to(device)
        kpc = torch.from_numpy(kp_coords).to(device)
        kpv = torch.from_numpy(kp_vis).to(device)
        kps = torch.from_numpy(kpt_score_map).to(device)

        xv, yv = torch.meshgrid(px_s, py_s)
        principal_samples = (
            torch.stack([xv, yv], dim=2).reshape(-1, 1, 2).repeat(1, kpc.shape[1], 1)
        )

        kpc = kpc.unsqueeze(1).repeat(1, principal_samples.shape[0], 1, 1)
        kpc += principal_samples
        kpc = kpc.reshape(-1, kpc.shape[2], 2)
        kpc = torch.round(kpc / down_sample_rate)

        kpv = kpv.unsqueeze(1).repeat(1, principal_samples.shape[0], 1)
        kpv = kpv.reshape(-1, kpv.shape[2])
        kpv[kpc[:, :, 0] < 0] = 0
        kpv[kpc[:, :, 0] >= hm_w - 1] = 0
        kpv[kpc[:, :, 1] < 0] = 0
        kpv[kpc[:, :, 1] >= hm_h - 1] = 0

        kpc[:, :, 0] = torch.clamp(kpc[:, :, 0], min=0, max=hm_w - 1)
        kpc[:, :, 1] = torch.clamp(kpc[:, :, 1], min=0, max=hm_h - 1)
        kpc = (kpc[:, :, 1:2] * hm_w + kpc[:, :, 0:1]).long()

        corr = torch.take_along_dim(kps.unsqueeze(0), kpc, dim=2)[:, :, 0]
        corr = torch.sum(corr * kpv, dim=1)

        all_corr.append(corr.reshape(-1, len(px_s), len(py_s)))

        begin += batch_size
        end += batch_size
        if end > len(px_samples):
            end = len(px_samples)

    corr = torch.cat(all_corr, dim=-2).detach().cpu().numpy()
    return corr


def solve_pose(
    cfg,
    feature_map,
    inter_module,
    kp_features,
    clutter_bank,
    poses,
    kp_coords,
    kp_vis,
    px_samples=None,
    py_samples=None,
    debug=False,
    device="cuda",
):
    nkpt, c = kp_features.size()
    memory = kp_features.view(nkpt, c, 1, 1)
    b, c, hm_h, hm_w = feature_map.size()
    pred = {}

    # Step 1: Pre-compute foreground and background features
    start_time = time.time()
    clutter_score = None
    if not isinstance(clutter_bank, list):
        clutter_bank = [clutter_bank]
    for cb in clutter_bank:
        _score = (
            torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3))
            .squeeze(0)
            .squeeze(0)
        )
        if clutter_score is None:
            clutter_score = _score
        else:
            clutter_score = torch.max(clutter_score, _score)

    kpt_score_map = torch.sum(
        feature_map.expand(nkpt, -1, -1, -1) * memory, dim=1
    )  # (nkpt, H, W)
    kpt_score_map = kpt_score_map.detach().cpu().numpy()
    kpt_score_map = kpt_score_map.reshape(nkpt, -1)  # (nkpt, H x W)

    end_time = time.time()
    pred["pre_compute_time"] = end_time - start_time

    # Step 2: Search for initializations
    start_time = end_time
    if px_samples is None or py_samples is None:
        if cfg.inference.search_translation:
            px_samples = np.linspace(
                0,
                hm_w * cfg.model.down_sample_rate,
                cfg.inference.num_px_samples,
                endpoint=True,
            )
            py_samples = np.linspace(
                0,
                hm_h * cfg.model.down_sample_rate,
                cfg.inference.num_py_samples,
                endpoint=True,
            )
        else:
            px_samples = np.array([hm_w * cfg.model.down_sample_rate / 2])
            py_samples = np.array([hm_h * cfg.model.down_sample_rate / 2])

    xv, yv = np.meshgrid(px_samples, py_samples, indexing="ij")
    corr = get_corr_pytorch(
        px_samples,
        py_samples,
        kpt_score_map,
        kp_coords,
        kp_vis,
        cfg.model.down_sample_rate,
        hm_h,
        hm_w,
        device=device,
    )
    corr = corr.reshape(
        poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3], len(xv), len(yv)
    )
    corr2d = corr.reshape(-1, len(xv), len(yv))

    corr2d_max = np.max(corr2d, axis=0)
    if cfg.inference.search_translation:
        extrema_2d = peak_local_max(
            corr2d_max, min_distance=cfg.inference.min_distance, exclude_border=False
        )
    else:
        extrema_2d = [[0, 0]]
    extrema = []
    for e in extrema_2d:
        c = corr2d[:, e[0], e[1]].reshape(
            poses.shape[0], poses.shape[1], poses.shape[2], poses.shape[3]
        )
        e_azim, e_elev, e_the, e_dist = np.unravel_index(
            np.argmax(c, axis=None), c.shape
        )
        if (
            not cfg.inference.search_translation
            or corr2d_max[e[0], e[1]] >= cfg.inference.pre_rendering_thr
        ):
            p = poses[e_azim, e_elev, e_the, e_dist]
            extrema.append(
                {
                    "azimuth": p[0],
                    "elevation": p[1],
                    "theta": p[2],
                    "distance": p[3],
                    "px": px_samples[e[0]],
                    "py": py_samples[e[1]],
                    "principal": [px_samples[e[0]], py_samples[e[1]]],
                }
            )
    if debug:
        pred["pre_rendering_poses"] = extrema
        pred["corr2d"] = corr2d_max

    if len(extrema) == 0:
        pred["final"] = []
        return pred

    end_time = time.time()
    pred["pre_rendering_time"] = end_time - start_time

    # Step 3: Refine object proposals with pose optimization
    start_time = end_time
    refined, object_score_list, seg_map_list = [], [], []
    for i in range(len(extrema)):
        C = camera_position_from_spherical_angles(
            extrema[i]["distance"],
            extrema[i]["elevation"],
            extrema[i]["azimuth"],
            degrees=False,
            device=device,
        )
        C = torch.nn.Parameter(C, requires_grad=True)
        theta = torch.tensor(extrema[i]["theta"], dtype=torch.float32).to(device)
        theta = torch.nn.Parameter(theta, requires_grad=True)
        max_principal = [extrema[i]["px"], extrema[i]["py"]]
        flow = torch.tensor(
            [
                -(max_principal[0] - hm_w * cfg.model.down_sample_rate / 2)
                / cfg.model.down_sample_rate
                * cfg.inference.translation_scale,
                -(max_principal[1] - hm_h * cfg.model.down_sample_rate / 2)
                / cfg.model.down_sample_rate
                * cfg.inference.translation_scale,
            ],
            dtype=torch.float32,
        ).to(device)
        flow = torch.nn.Parameter(flow, requires_grad=True)

        param_list = [C, theta]
        if cfg.inference.optimize_translation:
            param_list.append(flow)

        optim = construct_class_by_name(**cfg.inference.optimizer, params=param_list)
        scheduler_kwargs = {"optimizer": optim}
        scheduler = construct_class_by_name(
            **cfg.inference.scheduler, **scheduler_kwargs
        )

        for epo in range(cfg.inference.epochs):
            projected_map = inter_module(
                C,
                theta,
                mode=cfg.inference.inter_mode,
                blur_radius=cfg.inference.blur_radius,
            ).squeeze()
            flow_map = flow.view(1, 2, 1, 1).repeat(1, 1, hm_h, hm_w)
            projected_map = flow_warp(
                projected_map.unsqueeze(0), flow_map / cfg.inference.translation_scale
            )[0]
            object_score = torch.sum(projected_map * feature_map.squeeze(), dim=0)

            loss = call_func_by_name(
                func_name=cfg.inference.loss,
                obj_s=object_score,
                clu_s=clutter_score,
                device=device,
            )

            loss.backward()
            optim.step()
            optim.zero_grad()

        seg_map = (
            ((object_score > clutter_score) * (object_score > 0.0))
            .detach()
            .cpu()
            .numpy()
            .astype(np.uint8)
        )
        seg_map_list.append(seg_map)
        object_score_list.append(object_score.squeeze().detach().cpu().numpy())

        (
            distance_pred,
            elevation_pred,
            azimuth_pred,
        ) = camera_position_to_spherical_angle(C)
        theta_pred, distance_pred, elevation_pred, azimuth_pred = (
            theta.item(),
            distance_pred.item(),
            elevation_pred.item(),
            azimuth_pred.item(),
        )
        px_pred, py_pred = (
            -flow[0].item() / cfg.inference.translation_scale,
            -flow[1].item() / cfg.inference.translation_scale,
        )

        refined.append(
            {
                "azimuth": azimuth_pred,
                "elevation": elevation_pred,
                "theta": theta_pred,
                "distance": distance_pred,
                "px": px_pred,
                "py": py_pred,
                "principal": [
                    px_pred * cfg.model.down_sample_rate
                    + hm_w * cfg.model.down_sample_rate / 2,
                    py_pred * cfg.model.down_sample_rate
                    + hm_h * cfg.model.down_sample_rate / 2,
                ],
                "score": loss.item(),
            }
        )

    end_time = time.time()
    pred["optimization_time"] = end_time - start_time

    if len(refined) == 1:
        pred["final"] = refined
    else:
        raise NotImplementedError

    if debug:
        object_score_maps = np.array(object_score_list)
        segmentation_maps = np.array(seg_map_list)
        object_idx_map, new_seg_maps = resolve_occ(segmentation_maps, object_score_maps)
        pred["object_idx_map"] = object_idx_map
        pred["seg_maps"] = new_seg_maps

    return pred


def resolve_occ(seg_maps, score_maps):
    seg_maps_copy = copy.deepcopy(seg_maps)

    """
    obj_idx_map = np.zeros((seg_maps.shape[1], seg_maps.shape[2]), dtype=np.int16)
    obj_idx_map[seg_maps[0] == 1] = 1
    curr_score_map = copy.deepcopy(score_maps[0])
    for i in range(1, seg_maps.shape[0]):
        overlap = (obj_idx_map > 0) * (seg_maps[i] == 1)
    """

    occ_reasoning_mat = np.zeros((seg_maps.shape[0], seg_maps.shape[0]), dtype=np.int16)
    for i in range(0, seg_maps.shape[0]):
        for j in range(0, seg_maps.shape[0]):
            if i >= j:
                continue
            overlap = (seg_maps[i] == 1) * (seg_maps[j] == 1)
            if np.sum(overlap) == 0:
                continue
            score_i = (
                np.sum(overlap * score_maps[i])
                + np.sum((seg_maps[i] - overlap) * score_maps[i]) * 0.5
            )
            score_j = (
                np.sum(overlap * score_maps[j])
                + np.sum((seg_maps[j] - overlap) * score_maps[j]) * 0.5
            )
            if score_i >= score_j:
                occ_reasoning_mat[i][j] = 1
                occ_reasoning_mat[j][i] = -1
                seg_maps_copy[j][overlap == 1] = 0
            else:
                occ_reasoning_mat[j][i] = 1
                occ_reasoning_mat[i][j] = -1
                seg_maps_copy[i][overlap == 1] = 0

    sm = seg_maps_copy[0]
    for i in range(1, seg_maps_copy.shape[0]):
        sm += seg_maps_copy[i]
    # print(np.sum(sm > 1))
    # print([np.sum(seg_maps_copy[i] * score_maps[i]) for i in range(seg_maps_copy.shape[0])])

    seg_maps_pad = np.concatenate(
        [
            np.zeros(
                (1, seg_maps_copy.shape[1], seg_maps_copy.shape[2]),
                dtype=seg_maps_copy.dtype,
            ),
            seg_maps_copy,
        ],
        axis=0,
    )
    return np.argmax(seg_maps_pad, axis=0), seg_maps_copy


def pre_compute_kp_coords(
    mesh_path,
    azimuth_samples,
    elevation_samples,
    theta_samples,
    distance_samples,
    viewport=3000,
):
    """Calculate vertex visibility for cuboid models."""
    xvert, _ = load_off(mesh_path)

    xmin, xmax = np.min(xvert[:, 0]), np.max(xvert[:, 0])
    ymin, ymax = np.min(xvert[:, 1]), np.max(xvert[:, 1])
    zmin, zmax = np.min(xvert[:, 2]), np.max(xvert[:, 2])
    xmean = (xmin + xmax) / 2
    ymean = (ymin + ymax) / 2
    zmean = (zmin + zmax) / 2
    pts = [
        [xmean, ymean, zmin],
        [xmean, ymean, zmax],
        [xmean, ymin, zmean],
        [xmean, ymax, zmean],
        [xmin, ymean, zmean],
        [xmax, ymean, zmean],
    ]

    poses = np.zeros(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            4,
        ),
        dtype=np.float32,
    )
    num_vis_faces = []
    count = 0
    for azim_ in azimuth_samples:
        for elev_ in elevation_samples:
            for theta_ in theta_samples:
                for dist_ in distance_samples:
                    poses[count] = [azim_, elev_, theta_, dist_]
                    count += 1
                    if elev_ == 0:
                        if azim_ in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                            num_vis_faces.append(1)
                        else:
                            num_vis_faces.append(2)
                    else:
                        if azim_ in [0, np.pi / 2, np.pi, 3 * np.pi / 2]:
                            num_vis_faces.append(2)
                        else:
                            num_vis_faces.append(3)

    kp_coords = np.zeros(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            len(xvert),
            2,
        ),
        dtype=np.float32,
    )
    kp_vis = np.ones(
        (
            len(azimuth_samples)
            * len(elevation_samples)
            * len(theta_samples)
            * len(distance_samples),
            len(xvert),
        ),
        dtype=np.float32,
    )
    xvert_ext = np.concatenate((xvert, pts), axis=0)
    for i, pose_ in enumerate(poses):
        azim_, elev_, theta_, dist_ = pose_

        C = np.zeros((3, 1))
        C[0] = dist_ * math.cos(elev_) * math.sin(azim_)
        C[1] = -dist_ * math.cos(elev_) * math.cos(azim_)
        C[2] = dist_ * math.sin(elev_)
        azimuth = -azim_
        elevation = -(math.pi / 2 - elev_)
        Rz = np.array(
            [
                [math.cos(azimuth), -math.sin(azimuth), 0],
                [math.sin(azimuth), math.cos(azimuth), 0],
                [0, 0, 1],
            ]
        )  # rotation by azimuth
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(elevation), -math.sin(elevation)],
                [0, math.sin(elevation), math.cos(elevation)],
            ]
        )  # rotation by elevation
        R_rot = np.dot(Rx, Rz)
        R = np.hstack((R_rot, np.dot(-R_rot, C)))
        P = np.array([[viewport, 0, 0], [0, viewport, 0], [0, 0, -1]])
        x3d_ = np.hstack((xvert_ext, np.ones((len(xvert_ext), 1)))).T
        x3d_ = np.dot(R, x3d_)
        # x3d_r_ = np.dot(P, x3d_)
        x2d = np.dot(P, x3d_)
        x2d[0, :] = x2d[0, :] / x2d[2, :]
        x2d[1, :] = x2d[1, :] / x2d[2, :]
        x2d = x2d[0:2, :]
        R2d = np.array(
            [
                [math.cos(theta_), -math.sin(theta_)],
                [math.sin(theta_), math.cos(theta_)],
            ]
        )
        x2d = np.dot(R2d, x2d).T
        x2d[:, 1] *= -1

        # principal = np.array([px_, py_], dtype=np.float32)
        # x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

        x2d = x2d[: len(xvert)]
        kp_coords[i] = x2d

        center3d = x3d_[:, len(xvert) :]
        face_dist = np.sqrt(
            np.square(center3d[0, :])
            + np.square(center3d[1, :])
            + np.square(center3d[2, :])
        )
        ind = np.argsort(face_dist)[: num_vis_faces[i]]

        """
        # 13 13 5 13x12 + 13x12 + 5x12 + 5x12 + 5x13 + 5x13
        # 8 17 6 17x8 + 17x8 + 6x8 + 6x8 + 6x17 + 6x17
        if 0 in ind:
            kp_vis[i, 0:mesh_face_breaks[0]] = 1
        if 1 in ind:
            kp_vis[i, mesh_face_breaks[0]:mesh_face_breaks[1]] = 1
        if 2 in ind:
            kp_vis[i, mesh_face_breaks[1]:mesh_face_breaks[2]] = 1
        if 3 in ind:
            kp_vis[i, mesh_face_breaks[2]:mesh_face_breaks[3]] = 1
        if 4 in ind:
            kp_vis[i, mesh_face_breaks[3]:mesh_face_breaks[4]] = 1
        if 5 in ind:
            kp_vis[i, mesh_face_breaks[4]:mesh_face_breaks[5]] = 1
        """

    poses = poses.reshape(
        len(azimuth_samples),
        len(elevation_samples),
        len(theta_samples),
        len(distance_samples),
        4,
    )
    return poses, kp_coords, kp_vis
