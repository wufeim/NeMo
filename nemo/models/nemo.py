import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pytorch3d
from pytorch3d.structures import Meshes

from nemo.models.base_model import BaseModel
from nemo.models.feature_banks import mask_remove_near, remove_near_vertices_dist, StaticLatentMananger
from nemo.models.mesh_interpolate_module import MeshInterpolateModule
from nemo.models.solve_pose import pre_compute_kp_coords
from nemo.models.solve_pose import solve_pose
from nemo.models.batch_solve_pose import get_pre_render_samples
from nemo.models.batch_solve_pose import solve_pose as batch_solve_pose
from nemo.utils import center_crop_fun
from nemo.utils import construct_class_by_name
from nemo.utils import get_param_samples
from nemo.utils import normalize_features
from nemo.utils import pose_error, iou, pre_process_mesh_pascal, load_off
from nemo.utils.pascal3d_utils import IMAGE_SIZES, CATEGORIES

from nemo.models.project_kp import PackedRaster

from PIL import Image

def to_tensor(val):
    if isinstance(val, torch.Tensor):
        return val[None] if len(val.shape) == 2 else val
    elif isinstance(val, list):
        return [(t if isinstance(val, torch.Tensor) else torch.from_numpy(t)) for t in val]
    else:
        get = torch.from_numpy(val)
        return get[None] if len(get.shape) == 2 else get

class NeMo(BaseModel):
    def __init__(
        self,
        cfg,
        cate,
        mode,
        backbone,
        memory_bank,
        num_noise,
        max_group,
        down_sample_rate,
        mesh_path,
        training,
        inference,
        proj_mode='runtime',
        checkpoint=None,
        transforms=[],
        device="cuda:0",
        **kwargs
    ):
        super().__init__(cfg, cate, mode, checkpoint, transforms, ['loss', 'loss_main', 'loss_reg'], device)
        self.net_params = backbone
        self.memory_bank_params = memory_bank
        self.num_noise = num_noise
        self.max_group = max_group
        self.down_sample_rate = down_sample_rate
        self.training_params = training
        self.inference_params = inference
        self.dataset_config = cfg.dataset
        self.accumulate_steps = 0
        self.cate = cate
        self.batch_size = cfg.training.batch_size


        self.mesh_path = mesh_path.format(cate) if "{:s}" in mesh_path else mesh_path  
        self.category = CATEGORIES
        
        proj_mode = self.training_params.proj_mode
        
        if proj_mode != 'prepared':
            if self.training_params.classification:
                raster_conf = {
                            'image_size': self.dataset_config.image_sizes[self.category[0]],
                            **self.training_params.kp_projecter
                        }
                self.projector = []
                self.all_meshes = []
                self.all_vertices = []
                self.all_faces = []
                self.category = CATEGORIES
                self.all_verts_num, self.all_faces_num = [], []
                for subcate in self.category:
                    mesh_path_ = mesh_path.format(subcate) if "{:s}" in mesh_path else mesh_path           
                    mesh_ = pre_process_mesh_pascal(*load_off(mesh_path_, True)) 
                    self.all_vertices.append(mesh_[0].numpy())
                    self.all_faces.append(mesh_[1].numpy())
                    self.all_verts_num.append(mesh_[0].shape[0])
                    self.all_faces_num.append(mesh_[1].shape[0])   

                self.max_vert = max(self.all_verts_num)
                self.max_face = max(self.all_faces_num)
                self.build()
                if raster_conf['down_rate'] == -1:
                    raster_conf['down_rate'] = self.net.module.net_stride
                self.net.module.kwargs['n_vert'] = self.max_vert

                for i in range(len(self.all_vertices)):
                    vert_pad_size = self.max_vert - self.all_verts_num[i]
                    face_pad_size = self.max_face - self.all_faces_num[i]
                    self.all_vertices[i] = np.pad(self.all_vertices[i], pad_width=((0, vert_pad_size), (0, 0)), mode='constant', constant_values=0)
                    self.all_faces[i] = np.pad(self.all_faces[i], pad_width=((0, face_pad_size), (0, 0)), mode='constant', constant_values=-1)
                self.projector = PackedRaster(raster_conf, [self.all_vertices, self.all_faces], device='cuda')
                    
                '''
                mesh_ = Meshes(verts=to_tensor(mesh_[0]), faces=to_tensor(mesh_[1])).cuda()
                self.all_meshes.append(mesh_)
                aelf.projector.append(PackedRaster(raster_conf, mesh_, device='cuda'))
                '''
            else:
                self.build()
                raster_conf = {
                        'image_size': self.dataset_config.image_sizes[cate],
                        **self.training_params.kp_projecter
                    }
                if raster_conf['down_rate'] == -1:
                    raster_conf['down_rate'] = self.net.module.net_stride
                mesh_ = pre_process_mesh_pascal(*load_off(self.mesh_path, True))
                self.net.module.kwargs['n_vert'] = mesh_[0].shape[0]
                self.projector = PackedRaster(raster_conf, mesh_, device='cuda')
        else:
            self.projector = None

    def build(self):
        if self.mode == "train":
            self._build_train()
        else:
            self._build_inference()

    def _build_train(self):
        self.n_gpus = torch.cuda.device_count()
        if self.training_params.separate_bank:
            self.ext_gpu = f"cuda:{self.n_gpus-1}"
        else:
            self.ext_gpu = ""

        net = construct_class_by_name(**self.net_params)
        if self.training_params.separate_bank:
            self.net = nn.DataParallel(net, device_ids=[i for i in range(self.n_gpus - 1)]).cuda()
        else:
            self.net = nn.DataParallel(net).cuda()

        if self.training_params.classification:
            self.num_verts = self.max_vert
            memory_bank = construct_class_by_name(
                **self.memory_bank_params,
                output_size=self.num_verts*len(self.category)+self.num_noise*self.max_group,
                num_pos=self.num_verts*len(self.category),
                num_noise=self.num_noise,
                classification=True)
        else:
            self.num_verts = load_off(self.mesh_path)[0].shape[0]
            memory_bank = construct_class_by_name(
                **self.memory_bank_params,
                output_size=self.num_verts+self.num_noise*self.max_group,
                num_pos=self.num_verts,
                num_noise=self.num_noise)

        if self.training_params.separate_bank:
            self.memory_bank = memory_bank.cuda(self.ext_gpu)
        else:
            self.memory_bank = memory_bank.cuda()

        self.optim = construct_class_by_name(
            **self.training_params.optimizer, params=self.net.parameters())
        self.scheduler = construct_class_by_name(
            **self.training_params.scheduler, optimizer=self.optim)


        if self.training_params.classification:
            self.zeros = torch.zeros(self.batch_size, self.max_vert, self.max_vert * len(self.category), dtype=torch.float32).cuda()
        else:
            self.zeros = None

    def step_scheduler(self):
        self.scheduler.step()
        if self.projector is not None:
            if isinstance(self.projector, list):
                for projector in self.projector:
                    projector.step()
            else:
                self.projector.step()

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        if self.training_params.classification:
            index = sample["index"].cuda()
        else:
            index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()
        label = sample['label'].cuda()

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        
        if self.training_params.classification:
            kwargs_.update(dict(func_of_mesh=func_reselect, mesh_label=label))

        if self.training_params.proj_mode == 'prepared':
                kp = sample['kp'].cuda()
                kpvis = sample["kpvis"].cuda().type(torch.bool)
        else:
            with torch.no_grad():
                kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), img_label=label, **kwargs_)             
                if self.training_params.classification:
                    for i in range(kpvis.shape[0]):
                        kpvis[i, self.all_verts_num[label[i]]:] = False
            '''
            for i in range(kp.shape[0]):
                kp_t = kp[i].detach().cpu().numpy()
                kpvis_t = kpvis[i].detach().cpu().numpy()
                img2 = np.array(sample["original_img"][i])
                y0, y1, x0, x1, _, _ = sample["bbox"][i]
                obj_mask2 = sample["obj_mask"][i]
                
                import cv2

                for i2 in range(len(kp_t)):
                    if kpvis_t[i2]:
                        img2 = cv2.circle(
                            img2, (int(kp_t[i2, 1]), int(kp_t[i2, 0])), 2, (255, 0, 0), -1
                        )
                img2 = cv2.rectangle(img2, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

                gray_img = (img2 * 0.3).astype(np.uint8)
                gray_img[obj_mask2 == 1] = img2[obj_mask2 == 1]

                Image.fromarray(gray_img).save(
                    os.path.join('/home/guofeng/Classification_OmniNeMo/tmp', f'debug_{sample["this_name"][i].replace("/", "_")}.png')
                )
            '''
                                    
        features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True,)

        if self.training_params.separate_bank:
            get, y_idx, noise_sim = self.memory_bank(
                features.to(self.ext_gpu), index.to(self.ext_gpu), kpvis.to(self.ext_gpu), label.to(self.ext_gpu)
            )
        else:
            get, y_idx, noise_sim = self.memory_bank(features, index, kpvis, label)

        
        if not isinstance(self.projector, list) and self.projector is not None and 'voge' in self.projector.raster_type:
            kpvis = kpvis > self.projector.kp_vis_thr

        get /= self.training_params.T

        kappas={'pos':self.training_params.get('weight_pos', 0), 'near':self.training_params.get('weight_near', 1e5), 'clutter': -math.log(self.training_params.weight_noise)}
        # The default manner in VoGE-NeMo
        if self.training_params.remove_near_mode == 'vert':
            vert_ = self.projector.get_verts_recent()  # (B, K, 3)
            vert_dis = (vert_.unsqueeze(1) - vert_.unsqueeze(2)).pow(2).sum(-1).pow(.5)

            mask_distance_legal = remove_near_vertices_dist(
                vert_dis,
                thr=self.training_params.distance_thr,
                num_neg=self.num_noise * self.max_group,
                kappas=kappas,
            )
            if mask_distance_legal.shape[0] != get.shape[0]:
                mask_distance_legal = mask_distance_legal.expand(get.shape[0], -1, -1).contiguous()
        # The default manner in original-NeMo
        else:
            if self.zeros is not None and self.zeros.shape[0] != img.shape[0]:
                self.zeros = torch.zeros(img.shape[0], self.max_vert, self.max_vert * len(self.category), dtype=torch.float32).cuda()
            
            mask_distance_legal = mask_remove_near(
                kp,
                thr=self.training_params.distance_thr
                * torch.ones((img.shape[0],), dtype=torch.float32).cuda(),
                img_label=label,
                zeros=self.zeros,
                num_neg=self.num_noise * self.max_group,
                dtype_template=get,
                kappas=kappas,
            )

        if self.training_params.get('training_loss_type', 'nemo') == 'nemo':
            loss_main = nn.CrossEntropyLoss(reduction="none").cuda()(
                    (get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2]))[
                        kpvis.view(-1), :
                    ],
                    y_idx.view(-1)[kpvis.view(-1)],
            )
            loss_main = torch.mean(loss_main)
        elif self.training_params.get('training_loss_type', 'nemo') == 'kl_alan':
            loss_main = torch.mean((get.view(-1, get.shape[2]) * mask_distance_legal.view(-1, get.shape[2]))[kpvis.view(-1), :])

        if self.num_noise > 0:
            loss_reg = torch.mean(noise_sim) * self.training_params.loss_reg_weight
            loss = loss_main + loss_reg
        else:
            loss_reg = torch.zeros(1)
            loss = loss_main

        loss.backward()

        self.accumulate_steps += 1
        if self.accumulate_steps % self.training_params.train_accumulate == 0:
            self.optim.step()
            self.optim.zero_grad()

        self.loss_trackers['loss'].append(loss.item())
        self.loss_trackers['loss_main'].append(loss_main.item())
        self.loss_trackers['loss_reg'].append(loss_reg.item())

        return {'loss': loss.item(), 'loss_main': loss_main.item(), 'loss_reg': loss_reg.item()}

    def _build_inference(self):
        self.net = construct_class_by_name(**self.net_params)
        self.net = nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(self.checkpoint["state"])

        xvert, xface = load_off(self.mesh_path, to_torch=True)
        self.num_verts = int(xvert.shape[0])

        self.memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            output_size=self.num_verts,
            num_pos=self.num_verts,
            num_noise=0
        ).to(self.device)


        self.cate_index = CATEGORIES.index(self.cate)
        if self.training_params.classification:
            self.bank_base_index = self.cate_index * self.max_vert
            
            clutter = (
                self.checkpoint["memory"][len(CATEGORIES) * self.max_vert:]
                .detach()
                .cpu()
                .numpy()
            )
        else:
            self.bank_base_index = 0
            
            clutter = (
                self.checkpoint["memory"][self.memory_bank.memory.shape[0]:]
                .detach()
                .cpu()
                .numpy()
            )

        with torch.no_grad():
            self.memory_bank.memory.copy_(
                    self.checkpoint["memory"][self.bank_base_index : self.bank_base_index + self.memory_bank.memory.shape[0]]
            )
        memory = (
            self.checkpoint["memory"][self.bank_base_index : self.bank_base_index + self.memory_bank.memory.shape[0]]
            .detach()
            .cpu()
            .numpy()
        )
       
        self.feature_bank = torch.from_numpy(memory)
        self.clutter_bank = torch.from_numpy(clutter).to(self.device)
        self.clutter_bank = normalize_features(
            torch.mean(self.clutter_bank, dim=0)
        ).unsqueeze(0)
        self.kp_features = self.checkpoint["memory"][
            self.bank_base_index : self.bank_base_index + self.memory_bank.memory.shape[0]
        ].to(self.device)

        image_h, image_w = self.dataset_config.image_sizes[self.cate]
        # render_image_size = max(image_h, image_w) // self.down_sample_rate
        map_shape = (image_h // self.down_sample_rate, image_w // self.down_sample_rate)

        if self.inference_params.cameras.get('image_size', 0) == -1:
            self.inference_params.cameras['image_size'] = (map_shape, )
        if self.inference_params.cameras.get('principal_point', 0) == -1:
            self.inference_params.cameras['principal_point'] = ((map_shape[1] // 2, map_shape[0] // 2), )
        if self.inference_params.cameras.get('focal_length', None) is not None:
            self.inference_params.cameras['focal_length'] = self.inference_params.cameras['focal_length'] / self.down_sample_rate

        cameras = construct_class_by_name(**self.inference_params.cameras, device=self.device)
        raster_settings = construct_class_by_name(
            **self.inference_params.raster_settings, image_size=map_shape
        )
        if self.inference_params.rasterizer.class_name == 'VoGE.Renderer.GaussianRenderer':
            rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, render_settings=raster_settings
            )
        else:
            rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, raster_settings=raster_settings
            )
        self.inter_module = MeshInterpolateModule(
            xvert,
            xface,
            self.feature_bank,
            rasterizer=rasterizer,
            post_process=center_crop_fun(map_shape, (render_image_size,) * 2) if self.inference_params.get('center_crop', False) else None,
            convert_percentage=self.inference_params.get('convert_percentage', 0.5)
        ).to(self.device)

        (
            azimuth_samples,
            elevation_samples,
            theta_samples,
            distance_samples,
            px_samples,
            py_samples,
        ) = get_param_samples(self.cfg)

        self.init_mode = self.cfg.inference.get('init_mode', '3d_batch')

        if 'batch' in self.init_mode:
            dof = int(self.init_mode.split('d_')[0])
            self.feature_pre_rendered, self.cam_pos_pre_rendered, self.theta_pre_rendered = get_pre_render_samples(
                self.inter_module,
                azum_samples=azimuth_samples,
                elev_samples=elevation_samples,
                theta_samples=theta_samples,
                distance_samples=distance_samples,
                device=self.device
            )
            if dof == 3:
                assert distance_samples.shape[0] == 1
                self.record_distance = distance_samples[0]
            if dof == 6:
                self.samples_principal = torch.stack((torch.from_numpy(px_samples).view(-1, 1).expand(-1, py_samples.shape[0]) * map_shape[1], 
                                                  torch.from_numpy(py_samples).view(1, -1).expand(px_samples.shape[0], -1) * map_shape[0]), ).view(2, -1).T.to(self.device)

        else:
            self.poses, self.kp_coords, self.kp_vis = pre_compute_kp_coords(
                self.mesh_path,
                azimuth_samples=azimuth_samples,
                elevation_samples=elevation_samples,
                theta_samples=theta_samples,
                distance_samples=distance_samples,
            )

    def fast_inference(self, sample):
        self.net.eval()
        cls_index = self.cate_index

        n_vertice = self.num_verts
        img = sample["img"].to(self.device)
        score = {}

        with torch.no_grad():
            all_predicted_map = self.net.module.forward_test(img)

        B, C, H, W = all_predicted_map.shape
        all_similarity = []

        for i in range(B):
            predicted_map = all_predicted_map[i].view(C, -1)  # [C, H, W] -> [C, HW]

            object_score_per_vertex = torch.matmul(self.feature_bank.to(self.device), predicted_map)  # [N, C] x [C, HW] -> [N, HW]
            clutter_score = torch.matmul(self.clutter_bank, predicted_map).view(H, W)  # [H, W]

            object_score, vertex_activation_indices = object_score_per_vertex.max(dim=0)
            similarity = torch.maximum(object_score, clutter_score.view(-1, ))          

            similarity = similarity.view(H, W)  # [HW] -> [H, W]
            similarity_score = torch.sum(similarity) / (H * W)
            all_similarity.append(similarity_score)

        return all_similarity

            

    def evaluate(self, sample, debug=False):
        self.net.eval()

        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)
        if 'batch' in self.init_mode:
            dof = int(self.init_mode.split('d_')[0])
            if dof == 6:
                principal = self.samples_principal
            elif ('principal' in sample.keys()) and self.cfg.inference.get('realign', False):
                principal = sample['principal'].float().to(self.device) / self.down_sample_rate
            else:
                principal = None
            
            preds = batch_solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.clutter_bank,
                cam_pos_pre_rendered=self.cam_pos_pre_rendered,
                theta_pre_rendered=self.theta_pre_rendered,
                feature_pre_rendered=self.feature_pre_rendered,
                device=self.device,
                principal=principal,
                distance_source=sample['distance'].to(feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
                distance_target=self.record_distance * torch.ones(feature_map.shape[0]).to(feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
                pre_render=self.cfg.inference.get('pre_render', True),
                dof=dof,
            )
        else:
            assert len(img) == 1, "The batch size during validation should be 1"
            preds = solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.kp_features,
                self.clutter_bank,
                self.poses,
                self.kp_coords,
                self.kp_vis,
                debug=debug,
                device=self.device,
            )

        if isinstance(preds, dict):
            preds = [preds]
        
        to_print = []
        classification_result = {}
        for i, pred in enumerate(preds):
            if "azimuth" in sample and "elevation" in sample and "theta" in sample:
                pose_error_ = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]}, pred["final"][0])
                pred["pose_error"] = pose_error_
                if self.training_params.classification:
                    # print(pred['final'][0]['score'])
                    classification_result[sample['this_name'][i]] = (pred['final'][0]['score'], pose_error_)

                to_print.append(pose_error_)
        # print(to_print)
        return preds, classification_result

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt

    def predict_inmodal(self, sample, visualize=False):
        self.net.eval()

        # sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        assert len(img) == 1, "The batch size during validation should be 1"

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        clutter_score = None
        if not isinstance(self.clutter_bank, list):
            clutter_bank = [self.clutter_bank]
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

        nkpt, c = self.kp_features.shape
        feature_map_nkpt = feature_map.expand(nkpt, -1, -1, -1)
        kp_features = self.kp_features.view(nkpt, c, 1, 1)
        kp_score = torch.sum(feature_map_nkpt * kp_features, dim=1)
        kp_score, _ = torch.max(kp_score, dim=0)

        clutter_score = clutter_score.detach().cpu().numpy().astype(np.float32)
        kp_score = kp_score.detach().cpu().numpy().astype(np.float32)
        pred_mask = (kp_score > clutter_score).astype(np.uint8)
        pred_mask_up = cv2.resize(
            pred_mask, dsize=(pred_mask.shape[1]*self.down_sample_rate, pred_mask.shape[0]*self.down_sample_rate),
            interpolation=cv2.INTER_NEAREST)

        pred = {
            'clutter_score': clutter_score,
            'kp_score': kp_score,
            'pred_mask_orig': pred_mask,
            'pred_mask': pred_mask_up,
        }

        if 'inmodal_mask' in sample:
            gt_mask = sample['inmodal_mask'][0].detach().cpu().numpy()
            pred['gt_mask'] = gt_mask
            pred['iou'] = iou(gt_mask, pred_mask_up)

            obj_mask = sample['amodal_mask'][0].detach().cpu().numpy()
            pred['obj_mask'] = obj_mask

            # pred_mask_up[obj_mask == 0] = 0
            thr = 0.8
            new_mask = (kp_score > thr).astype(np.uint8)
            new_mask = cv2.resize(new_mask, dsize=(obj_mask.shape[1], obj_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            new_mask[obj_mask == 0] = 0
            pred['iou'] = iou(gt_mask, new_mask)
            pred['pred_mask'] = new_mask

        return pred

    def fix_init(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        if 'voge' in self.projector.raster_type:
            with torch.no_grad():
                frag_ = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)
  
            features, kpvis = self.net.forward(img, keypoint_positions=frag_, obj_mask=1 - obj_mask, do_normalize=True,)
        else:
            if self.training_params.proj_mode == 'prepared':
                kp = sample['kp'].cuda()
                kpvis = sample["kpvis"].cuda().type(torch.bool)
            else:
                with torch.no_grad():
                    kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)

            features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True,)
        return features.detach(), kpvis.detach()


class NeMoRuntimeOptimize(NeMo):
    def _build_train(self):
        super()._build_train()

        self.latent_iter_per = self.training_params.latent_iter_per
        self.latent_to_update = self.training_params.latent_to_update
        self.latent_lr = self.training_params.latent_lr

        self.n_shape_state = len(self.latent_to_update)
        self.latent_manager = StaticLatentMananger(n_latents=self.n_shape_state, to_device='cuda', store_device='cpu')
        self.latent_manager_optimizer = StaticLatentMananger(n_latents=self.n_shape_state * 2, to_device='cuda', store_device='cpu')

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        image_names = sample['this_name']
        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()

        feature_map_ = self.net.forward(img, do_normalize=True, mode=0)
        img_shape = img.shape[2::]

        # Dynamic implementation, for those in latent manager
        # params = self.latent_manager.get_latent(image_names, sample['params'].float())
        # Others
        # params = sample['params'].float()
        all_params = ['azimuth', 'elevation', 'distance', 'theta']

        get_values_updated = eval(
            'self.latent_manager.get_latent(image_names, ' +
            ', '.join(["sample['%s'].float()" % t for t in self.latent_to_update]) +
            ')'
            )
        azimuth, elevation, distance, theta = tuple([get_values_updated[self.latent_to_update.index(n_)] if n_ in self.latent_to_update else sample[n_].float() for n_ in all_params])

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        assert self.training_params.proj_mode != 'prepared'
        with torch.no_grad():
            kp, kpvis = self.projector(azim=azimuth.cuda(), elev=elevation.cuda(), dist=distance.cuda(), theta=theta.cuda(), **kwargs_)

        features = self.net.forward(feature_map_, keypoint_positions=kp, obj_mask=1 - obj_mask, img_shape=img_shape, mode=1)

        if self.training_params.separate_bank:
            get, y_idx, noise_sim = self.memory_bank(
                features.to(self.ext_gpu), index.to(self.ext_gpu), kpvis.to(self.ext_gpu)
            )
        else:
            get, y_idx, noise_sim = self.memory_bank(features, index, kpvis)
        
        if 'voge' in self.projector.raster_type:
            kpvis = kpvis > self.projector.kp_vis_thr

        get /= self.training_params.T

        kappas={'pos':self.training_params.get('weight_pos', 0), 'near':self.training_params.get('weight_near', 1e5), 'clutter': -math.log(self.training_params.weight_noise)}
        # The default manner in VoGE-NeMo
        if self.training_params.remove_near_mode == 'vert':
            vert_ = self.projector.get_verts_recent()  # (B, K, 3)
            vert_dis = (vert_.unsqueeze(1) - vert_.unsqueeze(2)).pow(2).sum(-1).pow(.5)

            mask_distance_legal = remove_near_vertices_dist(
                vert_dis,
                thr=self.training_params.distance_thr,
                num_neg=self.num_noise * self.max_group,
                kappas=kappas,
            )
            if mask_distance_legal.shape[0] != get.shape[0]:
                mask_distance_legal = mask_distance_legal.expand(get.shape[0], -1, -1).contiguous()
        # The default manner in original-NeMo
        else:
            mask_distance_legal = mask_remove_near(
                kp,
                thr=self.training_params.distance_thr
                * torch.ones((img.shape[0],), dtype=torch.float32).cuda(),
                num_neg=self.num_noise * self.max_group,
                dtype_template=get,
                kappas=kappas,
            )
        if self.training_params.get('training_loss_type', 'nemo') == 'nemo':
            loss_main = nn.CrossEntropyLoss(reduction="none").cuda()(
                (get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2]))[
                    kpvis.view(-1), :
                ],
                y_idx.view(-1)[kpvis.view(-1)],
            )
            loss_main = torch.mean(loss_main)
        elif self.training_params.get('training_loss_type', 'nemo') == 'kl_alan':
            loss_main = torch.mean((get.view(-1, get.shape[2]) * mask_distance_legal.view(-1, get.shape[2]))[kpvis.view(-1), :])

        self.optim.zero_grad()
        if self.num_noise > 0:
            loss_reg = torch.mean(noise_sim) * self.training_params.loss_reg_weight
            loss = loss_main + loss_reg
        else:
            loss_reg = torch.zeros(1)
            loss = loss_main

        loss.backward()
        self.optim.step()

        self.loss_trackers['loss'].append(loss.item())
        self.loss_trackers['loss_main'].append(loss_main.item())
        self.loss_trackers['loss_reg'].append(loss_reg.item())

        ############################ Optimize Pose During Training ##########################################
        feature_map_detached = feature_map_.detach()
        assert 'voge' in self.projector.raster_type, 'Current we only support VoGE as differentiable sampler.'

        params_dict = {'azimuth': azimuth, 'elevation': elevation, 'distance': distance, 'theta': theta}
        estep_params = [torch.nn.Parameter(params_dict[t]) for t in self.latent_to_update]
        optimizer_estep = torch.optim.Adam([{'params': t, } for t in estep_params], lr=self.latent_lr, betas=(0.8, 0.6))
        
        with torch.no_grad():
            states = self.latent_manager_optimizer.get_latent_without_default(image_names, )
            if states is not None:
                state_dicts_ = {'state': {i: {'step': 0, 'exp_avg': states[i * 2].clone(), 'exp_avg_sq': states[i * 2 + 1].clone()} for i in range(self.n_shape_state)}, 'param_groups': optimizer_estep.state_dict()['param_groups']}
                optimizer_estep.load_state_dict(state_dicts_)
            else:
                print('States not found in saved dict, this should only happens in epoch 0!')

        get_parameters = lambda param_name, params_dict=params_dict: estep_params[self.latent_to_update.index(param_name)] if param_name in self.latent_to_update else params_dict[param_name].cuda()
        all_loss_estep = []
        for latent_iter in range(self.latent_iter_per):
            optimizer_estep.zero_grad()
            kp, kpvis  = self.projector(azim=get_parameters('azimuth'), elev=get_parameters('elevation'), dist=get_parameters('distance'), theta=get_parameters('theta'), **kwargs_)
            features = self.net.forward(feature_map_detached, keypoint_positions=kp, obj_mask=1 - obj_mask, img_shape=img_shape, mode=1)

            loss_estep = self.memory_bank.compute_feature_dist(features, kpvis)
            loss_estep.backward()

            optimizer_estep.step()
            all_loss_estep.append(loss_estep.item())

        optimizer_estep.zero_grad()
        with torch.no_grad():
            state_dict_optimizer = optimizer_estep.state_dict()['state']
            self.latent_manager_optimizer.save_latent(image_names, *[state_dict_optimizer[i // 2]['exp_avg_sq' if i % 2 else 'exp_avg'] for i in range(self.n_shape_state * 2)])
            self.latent_manager.save_latent(image_names, *estep_params)

        return {'loss': loss.item(), 'loss_main': loss_main.item(), 'loss_reg': loss_reg.item(), 'loss_estep': sum(all_loss_estep)}

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        ckpt['update_latents'] = self.latent_to_update
        ckpt['latents'] = self.latent_manager.latent_set
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt
