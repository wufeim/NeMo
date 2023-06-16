# Author: Angtian Wang
# Adding support for batch operation 
# Perform in original NeMo manner 
# Support 3D pose as NeMo and VoGE-NeMo, 
# Not support 6D pose in current version


import numpy as np
import torch
from pytorch3d.renderer import camera_position_from_spherical_angles
from nemo.utils import construct_class_by_name
from nemo.utils import camera_position_to_spherical_angle
import time

try:
    from VoGE.Renderer import GaussianRenderer, GaussianRenderSettings, interpolate_attr
    from VoGE.Utils import Batchifier
    enable_voge = True
except:
    enable_voge=False

if not enable_voge:
    from TorchBatchifier import Batchifier


def loss_fg_only(obj_s, clu_s=None, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - reduce_method(obj_s)


def loss_fg_bg(obj_s, clu_s, reduce_method=lambda x: torch.mean(x)):
    return torch.ones(1, device=obj_s.device) - (
        reduce_method(torch.max(obj_s, clu_s)) - reduce_method(clu_s)
    )


def get_pre_render_samples(inter_module, azum_samples, elev_samples, theta_samples, set_distance=5, device='cpu',):
    with torch.no_grad():
        get_c = []
        get_theta = []
        get_samples = [[azum_, elev_, theta_] for azum_ in azum_samples for elev_ in elev_samples for theta_ in theta_samples]
        out_maps = []
        for sample_ in get_samples:
            theta_ = torch.ones(1, device=device) * sample_[2]
            C = camera_position_from_spherical_angles(set_distance, sample_[1], sample_[0], degrees=False, device=device)

            projected_map = inter_module(C, theta_)
            out_maps.append(projected_map)
            get_c.append(C.detach())
            get_theta.append(theta_)

        get_c = torch.stack(get_c, ).squeeze(1)
        get_theta = torch.cat(get_theta)
        out_maps = torch.stack(out_maps)

    return out_maps, get_c, get_theta


def get_init_pos_rendered(samples_maps, samples_pos, samples_theta, predicted_maps, clutter_scores=None, batch_size=32):
    """
    samples_pos: [n, 3]
    samples_theta: [n, ]
    samples_map: [n, c, h, w]
    predicted_map: [b, c, h, w]
    clutter_score: [b, h, w]
    """
    n = samples_maps.shape[0]
    if clutter_scores is None:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    batchifier = Batchifier(batch_size=batch_size, batch_args=('projected_map', 'predicted_map') + (tuple() if clutter_scores is None else ('clutter_map', )), target_dims=(0, 1))

    with torch.no_grad():
        # [n, b, c, h, w] -> [n, b]
        target_shape = (n, *predicted_maps.shape)
        get_loss = batchifier(cal_sim)(projected_map=samples_maps.expand(*target_shape).contiguous(),
                                       predicted_map=predicted_maps[None].expand(*target_shape).contiguous(), 
                                       clutter_map=clutter_scores[None].expand(n, *clutter_scores.shape).contiguous(), )

        # [b]
        use_indexes = torch.min(get_loss, dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes)
    # return samples_pos[use_indexes], samples_theta[use_indexes]


def solve_pose(
    cfg,
    feature_map,
    inter_module,
    clutter_bank,
    cam_pos_pre_rendered,
    theta_pre_rendered,
    feature_pre_rendered,
    device="cuda",
):
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
            .squeeze(1)
        )
        if clutter_score is None:
            clutter_score = _score
        else:
            clutter_score = torch.max(clutter_score, _score)

    end_time = time.time()
    pred["pre_compute_time"] = end_time - start_time

    # Step 2: Search for initializations
    start_time = end_time
    init_C, init_theta = get_init_pos_rendered(samples_maps=feature_pre_rendered, 
                                               samples_pos=cam_pos_pre_rendered, 
                                               samples_theta=theta_pre_rendered, 
                                               predicted_maps=feature_map, 
                                               clutter_scores=clutter_score, 
                                               batch_size=cfg.get('batch_size_no_grad', 64))
    end_time = time.time()
    pred["pre_rendering_time"] = end_time - start_time
    
    # Step 3: Refine object proposals with pose optimization
    start_time = end_time

    C = torch.nn.Parameter(init_C, requires_grad=True)
    theta = torch.nn.Parameter(init_theta, requires_grad=True)
    optim = construct_class_by_name(**cfg.inference.optimizer, params=[C, theta])

    scheduler_kwargs = {"optimizer": optim}
    scheduler = construct_class_by_name(**cfg.inference.scheduler, **scheduler_kwargs)
    
    for epo in range(cfg.inference.epochs):
        # [b, c, h, w]
        projected_map = inter_module(
            C,
            theta,
            mode=cfg.inference.inter_mode,
            blur_radius=cfg.inference.blur_radius,
        )

        # [b, c, h, w] -> [b, h, w]
        object_score = torch.sum(projected_map * feature_map, dim=1)
        
        loss = loss_fg_bg(object_score, clutter_score, )

        loss.backward()
        optim.step()
        optim.zero_grad()

        if (epo + 1) % (cfg.inference.epochs // 3) == 0:
            scheduler.step()

    distance_preds, elevation_preds, azimuth_preds = camera_position_to_spherical_angle(C)
    pred["optimization_time"] = end_time - start_time

    preds = []

    for i in range(b):
        theta_pred, distance_pred, elevation_pred, azimuth_pred = (
            theta[i].item(),
            distance_preds[i].item(),
            elevation_preds[i].item(),
            azimuth_preds[i].item(),
        )
        with torch.no_grad():
            this_loss = loss_fg_bg(object_score[i, None], clutter_score[i, None], )
        refined = [{
                "azimuth": azimuth_pred,
                "elevation": elevation_pred,
                "theta": theta_pred,
                "distance": distance_pred,
                "principal": [
                    hm_w * cfg.model.down_sample_rate / 2,
                    hm_h * cfg.model.down_sample_rate / 2,
                ],
                "score": this_loss.item(),}]
        preds.append(dict(final=refined, **{k: pred[k] / b for k in pred.keys()}))

    return preds