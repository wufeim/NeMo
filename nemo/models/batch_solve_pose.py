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
from nemo.utils.general import tensor_linspace
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


def get_pre_render_samples(inter_module, azum_samples, elev_samples, theta_samples, distance_samples=[5], device='cpu',):
    with torch.no_grad():
        get_c = []
        get_theta = []
        get_samples = [[azum_, elev_, theta_, distance_] for azum_ in azum_samples for elev_ in elev_samples for theta_ in theta_samples for distance_ in distance_samples]
        out_maps = []
        for sample_ in get_samples:
            theta_ = torch.ones(1, device=device) * sample_[2]
            C = camera_position_from_spherical_angles(sample_[3], sample_[1], sample_[0], degrees=False, device=device)

            projected_map = inter_module(C, theta_)
            out_maps.append(projected_map)
            get_c.append(C.detach())
            get_theta.append(theta_)

        get_c = torch.stack(get_c, ).squeeze(1)
        get_theta = torch.cat(get_theta)
        out_maps = torch.stack(out_maps)

    return out_maps, get_c, get_theta


@torch.no_grad()
def align_no_centered(maps_source, distance_source, principal_source, maps_target_shape, distance_target, principal_target, padding_mode='zeros'):
    """
    maps_source: [n, c, h1, w1]
    distance_source: [n, ]
    principal_source: [n, 2]
    """
    n, c, h1, w1 = maps_source.shape
    h0, w0 = maps_target_shape

    # distance source larger, sampling grid wider
    resize_rate = (distance_source / distance_target).float()

    range_x_min = 2 * principal_source[:, 0] / w1 - w0 / (w1 * resize_rate) - principal_target[:, 0] * 2 / w0
    range_x_max = 2 * principal_source[:, 0] / w1 + w0 / (w1 * resize_rate) - principal_target[:, 0] * 2 / w0
    range_y_min = 2 * principal_source[:, 1] / h1 - h0 / (h1 * resize_rate) - principal_target[:, 1] * 2 / h0
    range_y_max = 2 * principal_source[:, 1] / h1 + h0 / (h1 * resize_rate) - principal_target[:, 1] * 2 / h0

    # [n, w0] -> [n, h0, w0]
    grid_x = tensor_linspace(range_x_min, range_x_max, int(w0.item()))[:, None, :].expand(-1, int(h0.item()), -1)
    # [n, h0] -> [n, h0, w0]
    grid_y = tensor_linspace(range_y_min, range_y_max, int(h0.item()))[:, :, None].expand(-1, -1, int(w0.item()))

    grids = torch.cat([grid_x[..., None], grid_y[..., None]], dim=3)

    return torch.nn.functional.grid_sample(maps_source, grids, padding_mode=padding_mode)


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
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(get_loss, dim=0)[0]


def get_init_pos_rendered_dim0(samples_maps, samples_pos, samples_theta, predicted_maps, clutter_scores=None, batch_size=32):
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
            object_score = torch.einsum('nchw,chw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,chw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    batchifier = Batchifier(batch_size=batch_size, batch_args=('projected_map', ), target_dims=(0, ))

    with torch.no_grad():
        # [n, b, c, h, w] -> [n, b]
        get_loss = []

        for i in range(predicted_maps.shape[0]):
            # [n]
            get_loss.append(batchifier(cal_sim)(projected_map=samples_maps.squeeze(1), predicted_map=predicted_maps[i], clutter_map=clutter_scores[None, i]))

        # b * [n, ] -> [n, b]
        get_loss = torch.stack(get_loss).T

        # [b]
        use_indexes = torch.min(get_loss, dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(get_loss, dim=0)[0]



def get_init_pos(inter_module, samples_pos, samples_theta, predicted_maps, clutter_scores=None, reset_distance=None):
    if clutter_scores is None:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_only(object_score, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    else:
        def cal_sim(projected_map, predicted_map, clutter_map):
            object_score = torch.einsum('nchw,nchw->nhw', projected_map, predicted_map)
            return loss_fg_bg(object_score, clutter_map, reduce_method=lambda x: torch.mean(x, dim=(1, 2)))

    with torch.no_grad():
        out_scores = []
        for pos_, theta_ in zip(samples_pos, samples_theta):
            if reset_distance is not None:
                maps_ = inter_module(torch.nn.functional.normalize(pos_[None]) * reset_distance[:, None], theta_[None].expand(reset_distance.shape[0], -1))
            else:
                maps_ = inter_module(pos_[None], theta_[None])
            scores_ = cal_sim(maps_, predicted_maps, clutter_scores)
            out_scores.append(scores_)
        use_indexes = torch.min(torch.stack(out_scores), dim=0)[1]

    # [n, 3] -> [b, 3]
    return torch.gather(samples_pos, dim=0, index=use_indexes.view(-1, 1).expand(-1, 3)), torch.gather(samples_theta, dim=0, index=use_indexes), torch.min(torch.stack(out_scores), dim=0)[0]


def solve_pose(
    cfg,
    feature_map,
    inter_module,
    clutter_bank,
    cam_pos_pre_rendered,
    theta_pre_rendered,
    feature_pre_rendered,
    device="cuda",
    principal=None,
    pre_render=True,
    dof=3,
    **kwargs
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

    # 3 DoF or 4 DoF
    if dof == 3 or dof == 4:
        # Not centered images
        if principal is not None:
            maps_target_shape = inter_module.rasterizer.cameras.image_size 
            t_feature_map = align_no_centered(maps_source=feature_map, principal_source=principal, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, **kwargs)
            t_clutter_score = align_no_centered(maps_source=clutter_score[:, None], principal_source=principal, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, **kwargs).squeeze(1)
            init_principal = principal.float()
            inter_module.rasterizer.cameras._N = feature_map.shape[0]
        # Centered images
        else:
            init_principal = inter_module.rasterizer.cameras.principal_point
            t_feature_map = feature_map
            t_clutter_score = clutter_score

        if pre_render:
            init_C, init_theta, _ = get_init_pos_rendered_dim0(samples_maps=feature_pre_rendered, 
                                                    samples_pos=cam_pos_pre_rendered, 
                                                    samples_theta=theta_pre_rendered, 
                                                    predicted_maps=t_feature_map, 
                                                    clutter_scores=t_clutter_score, 
                                                    batch_size=cfg.get('batch_size_no_grad', 144))
        else:
            init_C, init_theta, _ = get_init_pos(inter_module=inter_module, 
                                                    samples_pos=cam_pos_pre_rendered, 
                                                    samples_theta=theta_pre_rendered, 
                                                    predicted_maps=feature_map, 
                                                    clutter_scores=clutter_score, 
                                                    reset_distance=kwargs.get('distance_source').float())

    # 6 DoF
    else:
        assert pre_render
        maps_target_shape = inter_module.rasterizer.cameras.image_size 

        with torch.no_grad():
            all_init_C, all_init_theta, all_init_loss = [], [], []
            for principal_ in principal:
                n = feature_map.shape[0]
                distance_source = kwargs.get('distance_source')
                principal_ = principal_[None].expand(n, -1).float()
                t_feature_map = align_no_centered(maps_source=feature_map, principal_source=principal_, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_source, padding_mode='border')
                t_clutter_score = align_no_centered(maps_source=clutter_score[:, None], principal_source=principal_, maps_target_shape=(maps_target_shape[0, 0], maps_target_shape[0, 1]), principal_target=maps_target_shape.flip(1) / 2, distance_source=distance_source, distance_target=distance_source, padding_mode='border').squeeze(1)

                this_C, this_theta, this_loss = get_init_pos_rendered_dim0(samples_maps=feature_pre_rendered, 
                                                        samples_pos=cam_pos_pre_rendered, 
                                                        samples_theta=theta_pre_rendered, 
                                                        predicted_maps=t_feature_map, 
                                                        clutter_scores=t_clutter_score, 
                                                        batch_size=cfg.get('batch_size_no_grad', 144),)
                
                all_init_C.append(this_C)
                all_init_theta.append(this_theta)
                all_init_loss.append(this_loss)

            use_indexes = torch.min(torch.stack(all_init_loss), dim=0)[1]
            init_C = torch.gather(torch.stack(all_init_C), dim=0, index=use_indexes.view(1, -1, 1).expand(-1, -1, 3)).squeeze(0)
            init_theta = torch.gather(torch.stack(all_init_theta), dim=0, index=use_indexes.view(1, -1)).squeeze(0)
            init_principal = torch.gather(principal, dim=0, index=use_indexes.view(-1, 1).expand(-1, 2)).float()
            
            inter_module.rasterizer.cameras._N = feature_map.shape[0]

    end_time = time.time()
    pred["pre_rendering_time"] = end_time - start_time

    # Step 3: Refine object proposals with pose optimization
    start_time = end_time
    
    if principal is not None and dof == 3:
        init_C = init_C / init_C.pow(2).sum(-1).pow(.5)[..., None] * kwargs.get('distance_source')[..., None].float()

    C = torch.nn.Parameter(init_C, requires_grad=True)
    theta = torch.nn.Parameter(init_theta, requires_grad=True)
    if dof == 6 or cfg.get('optimize_translation', False):
        principals = torch.nn.Parameter(init_principal, requires_grad=True)
        inter_module.rasterizer.cameras.principal_point = principals
        optim = construct_class_by_name(**cfg.inference.optimizer, params=[C, theta, principals])
    else:
        principals = init_principal.expand(b, -1) if init_principal.shape[0] == 1 else init_principal
        inter_module.rasterizer.cameras.principal_point = init_principal
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
        this_principal = principals[i]
        with torch.no_grad():
            this_loss = loss_fg_bg(object_score[i, None], clutter_score[i, None], )
        refined = [{
                "azimuth": azimuth_pred,
                "elevation": elevation_pred,
                "theta": theta_pred,
                "distance": distance_pred,
                "principal": [
                    this_principal[0].item(),
                    this_principal[1].item(),
                ],
                "score": this_loss.item(),}]
        preds.append(dict(final=refined, **{k: pred[k] / b for k in pred.keys()}))

    return preds
