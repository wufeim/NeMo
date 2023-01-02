import torch
import torch.nn as nn

from nemo.models.base_model import BaseModel
from nemo.models.feature_banks_cls import mask_remove_near
from nemo.models.mesh_interpolate_module import MeshInterpolateModule
from nemo.models.solve_pose import pre_compute_kp_coords
from nemo.models.solve_pose import solve_pose
from nemo.utils import center_crop_fun
from nemo.utils import construct_class_by_name
from nemo.utils import get_param_samples
from nemo.utils import load_off
from nemo.utils import normalize_features
from nemo.utils import pose_error
from nemo.utils.pascal3d_utils import CATEGORIES
from nemo.utils.pascal3d_utils import IMAGE_SIZES


class NeMoCls(BaseModel):
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
        image_sizes,
        training,
        inference,
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
        self.mesh_path = mesh_path
        self.image_sizes = image_sizes
        self.training_params = training
        self.inference_params = inference

        self.accumulate_steps = 0

        self.build()

    def build(self):
        self.sanity_check()
        if self.mode == "train":
            self._build_train()
        else:
            self._build_inference()

    def sanity_check(self):
        # All image sizes should be the same
        size_0 = self.image_sizes['aeroplane']
        for cate in CATEGORIES:
            assert size_0 == self.image_sizes[cate], \
                "All image sizes should be the same for classification"

    def _build_train(self):
        self.n_gpus = torch.cuda.device_count()
        if self.training_params.separate_bank:
            self.ext_gpu = f"cuda:{self.n_gpus-1}"
        else:
            self.ext_gpu = ""

        self.all_num_verts = []
        self.all_mesh_paths = []
        self.pad_index = []

        assert '{:s}' in self.mesh_path, 'The mesh path should contain {:s} to format paths with categories'
        for cate in CATEGORIES:
            m_p = self.mesh_path.format(cate)
            self.all_mesh_paths.append(m_p)
            self.all_num_verts.append(load_off(m_p)[0].shape[0])

        net = construct_class_by_name(**self.net_params)
        if self.training_params.separate_bank:
            self.net = nn.DataParallel(net, device_ids=[i for i in range(self.n_gpus - 1)]).cuda()
        else:
            self.net = nn.DataParallel(net).cuda()

        memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            output_size=len(CATEGORIES)*max(self.all_num_verts)+self.num_noise*self.max_group,
            num_pos=len(CATEGORIES)*max(self.all_num_verts),
            n_list_set=self.all_num_verts,
            num_noise=self.num_noise)
        if self.training_params.separate_bank:
            self.memory_bank = memory_bank.cuda(self.ext_gpu)
        else:
            self.memory_bank = memory_bank.cuda()

        self.optim = construct_class_by_name(
            **self.training_params.optimizer, params=self.net.parameters())
        self.scheduler = construct_class_by_name(
            **self.training_params.scheduler, optimizer=self.optim)

        for i in range(len(CATEGORIES)):
            n = max(self.all_num_verts) - self.all_num_verts[i]
            for j in range(n):
                self.pad_index.append(max(self.all_num_verts) * i + self.all_num_verts[i] + j)
        self.pad_index = torch.tensor(self.pad_index, dtype=torch.long)

    def _build_inference(self):
        self.all_num_verts = []
        assert '{:s}' in self.mesh_path, 'The mesh path should contain {:s} to format paths with categories'
        for cate in CATEGORIES:
            m_p = self.mesh_path.format(cate)
            self.all_num_verts.append(load_off(m_p)[0].shape[0])

        net = construct_class_by_name(**self.net_params)
        self.net = nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(self.checkpoint["state"])

        self.memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            output_size=len(CATEGORIES)*max(self.all_num_verts)+self.num_noise*self.max_group,
            num_pos=len(CATEGORIES)*max(self.all_num_verts),
            n_list_set=self.all_num_verts,
            num_noise=self.num_noise
        ).to(self.device)
        with torch.no_grad():
            self.memory_bank.memory.copy_(checkpoint['memory'][0:memory_bank.memory.shape[0]])

        image_h, image_w = self.image_sizes['aeroplane']
        render_image_size = max(image_h, image_w) // self.down_sample_rate
        map_shape = (image_h // self.down_sample_rate, image_w // self.down_sample_rate)

        cameras = construct_class_by_name(
            class_name=self.inference_params.cameras.class_name,
            focal_length=self.inference_params.cameras.focal_length // self.down_sample_rate,
            principal_point=((map_shape[1] // 2, map_shape[0] // 2),),
            image_size=(map_shape,),
            in_ndc=False,
            device=self.device)
        raster_settings = construct_class_by_name(
            **self.inference_params.raster_settings, image_size=render_image_size
        )
        rasterizer = construct_class_by_name(
            **self.inference_params.rasterizer, cameras=cameras, raster_settings=raster_settings
        )

        clutter = self.checkpoint["memory"][self.memory_bank.memory.shape[0] :].detach().cpu().numpy()
        self.clutter_bank = torch.from_numpy(clutter).to(self.device)
        self.clutter_bank = normalize_features(torch.mean(self.clutter_bank, dim=0)).unsqueeze(0)

        self.inter_module, self.kp_features = {}, {}
        for idx, cate in enumerate(CATEGORIES):
            memory = self.checkpoint["memory"][idx*max(self.all_num_verts):(idx+1)*max(self.all_num_verts)].detach().cpu().numpy()
            feature_bank = torch.from_numpy(memory)
            self.kp_features[cate] = self.checkpoint["memory"][idx*max(self.all_num_verts):(idx+1)*max(self.all_num_verts)].to(self.device)
            xvert, xface = load_off(self.mesh_path.format(cate), to_torch=True)
            self.inter_module[cate] = MeshInterpolateModule(
                xvert,
                xface,
                feature_bank,
                rasterizer,
                post_process=center_crop_fun(map_shape, (render_image_size,) * 2),
            )

        (
            azimuth_samples,
            elevation_samples,
            theta_samples,
            distance_samples,
            px_samples,
            py_samples,
        ) = get_param_samples(self.cfg)

        self.poses, self.kp_coords, self.kp_vis = {}, {}, {}
        for cate in CATEGORIES:
            self.poses[cate], self.kp_coords[cate], self.kp_vis[cate] = pre_compute_kp_coords(
                self.mesh_path,
                azimuth_samples=azimuth_samples,
                elevation_samples=elevation_samples,
                theta_samples=theta_samples,
                distance_samples=distance_samples,
            )

    def step_scheduler(self):
        self.scheduler.step()

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        kp = sample['kp'].cuda()
        kpvis = sample["kpvis"].cuda().type(torch.bool)
        obj_mask = sample["obj_mask"].cuda()
        index = sample["index"].cuda()
        label = sample["label"].cuda()

        features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask)

        if self.training_params.separate_bank:
            get, y_idx, noise_sim, label_onehot = self.memory_bank(
                features.to(self.ext_gpu), index.to(self.ext_gpu), kpvis.to(self.ext_gpu),
                label.to(self.ext_gpu)
            )
        else:
            get, y_idx, noise_sim, label_onehot = self.memory_bank(features, index, kpvis, label)

        get /= self.training_params.T

        mask_distance_legal = mask_remove_near(
            kp,
            thr=self.training_params.distance_thr,
            num_neg=self.num_noise * self.max_group,
            img_label=label,
            n_list=self.all_num_verts,
            pad_index=self.pad_index,
            zeros=torch.zeros(img.shape[0], max(self.all_num_verts), max(self.all_num_verts) * len(CATEGORIES), dtype=torch.float32).to(get.device),
            dtype_template=get,
            neg_weight=self.training_params.weight_noise,
        )

        loss_main = nn.CrossEntropyLoss(reduction="none").cuda()(
            get.view(-1, get.shape[2])[kpvis.view(-1), :], index.view(-1)[kpvis.view(-1)]
        )
        loss_main = torch.mean(loss_main)

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

    def evaluate(self, sample, debug=False):
        self.net.eval()

        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        assert len(img) == 1, "The batch size during validation should be 1"

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        scores, preds = [], []
        for cate in CATEGORIES:
            self.inter_module[cate].to(self.device)
            pred = solve_pose(
                self.cfg,
                feature_map,
                self.inter_module[cate],
                self.kp_features[cate],
                self.clutter_bank,
                self.poses[cate],
                self.kp_coords[cate],
                self.kp_vis[cate],
                debug=debug,
                device=self.device,
            )
            self.inter_module[cate].to('cpu')
            preds.append(pred)
            scores.append(pred['final'][0]['score'])

        pred_cate = np.argmin(scores)
        pred = preds[pred_cate]

        if "azimuth" in sample and "elevation" in sample and "theta" in sample:
            pred["pose_error"] = pose_error(sample, pred["final"][0])

        return pred

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt
