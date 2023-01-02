import numpy as np
import torch
import torch.nn as nn
import torchvision

from nemo.models.base_model import BaseModel
from nemo.utils import construct_class_by_name


class ResNetGeneral(BaseModel):
    def __init__(self, cfg, cate, mode, output_dim, num_bins, backbone, training, checkpoint=None, transforms=[], device='cuda:0', **kwargs):
        super().__init__(cfg, cate, mode, checkpoint, transforms, ['loss'], device)
        self.output_dim = output_dim
        self.num_bins = num_bins
        self.backbone = backbone
        self.training = training

        self.build()

    def build(self):
        assert self.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], \
            f"Unsupported backbone {self.backbone} for ResNetGeneral"

        model = torchvision.models.__dict__[self.backbone](pretrained=True)
        model.avgpool = nn.AvgPool2d(8, stride=1)
        if self.backbone == 'resnet18':
            model.fc = nn.Linear(512 * 1, self.output_dim)
        else:
            model.fc = nn.Linear(512 * 4, self.output_dim)
        self.model = model.to(self.device)
        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['state'])

        self.loss_func = construct_class_by_name(**self.training.loss).to(self.device)
        self.optim = construct_class_by_name(
            **self.training.optimizer, params=self.model.parameters())
        self.scheduler = construct_class_by_name(
            **self.training.scheduler, optimizer=self.optim)

    def train(self, sample):
        self.model.train()
        sample = self.transforms(sample)

        img = sample['img'].to(self.device)
        targets = self._get_targets(sample).long().view(-1).to(self.device)
        output = self.model(img)

        loss = construct_class_by_name(**self.training.loss).to(self.device)(
            output.view(-1, self.num_bins), targets)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        self.loss_trackers['loss'].append(loss.item())

        return {'loss': loss.item()}

    def step_scheduler(self):
        self.scheduler.step()

    def _get_targets(self, sample):
        azimuth = sample['azimuth'].numpy() / np.pi
        elevation = sample['elevation'].numpy() / np.pi
        theta = sample['theta'].numpy() / np.pi
        theta[theta < -1.0] += 2.0
        theta[theta > 1.0] -= 2.0

        targets = np.zeros((len(azimuth), 3), dtype=np.int32)
        targets[azimuth < 0.0, 0] = self.num_bins - 1 - np.floor(-azimuth[azimuth < 0.0] * self.num_bins / 2.0)
        targets[azimuth >= 0.0, 0] = np.floor(azimuth[azimuth >= 0.0] * self.num_bins / 2.0)
        targets[:, 1] = np.ceil(elevation * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)
        targets[:, 2] = np.ceil(theta * self.num_bins / 2.0 + self.num_bins / 2.0 - 1)

        return torch.from_numpy(targets)

    def _prob_to_pose(self, prob):
        pose_pred = np.argmax(prob.reshape(-1, 3, self.num_bins), axis=2).astype(np.float32)
        pose_pred[:, 0] = (pose_pred[:, 0] + 0.5) * np.pi / (self.num_bins / 2.0)
        pose_pred[:, 1] = (pose_pred[:, 1] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
        pose_pred[:, 2] = (pose_pred[:, 2] - self.num_bins / 2.0) * np.pi / (self.num_bins / 2.0)
        return pose_pred

    def evaluate(self, sample):
        self.model.eval()
        sample = self.transforms(sample)

        img = sample['img'].to(self.device)
        output = self.model(img).detach().cpu().numpy()

        img_flip = torch.flip(img, dims=[3])
        output_flip = self.model(img_flip).detach().cpu().numpy()

        azimuth = output_flip[:, :self.num_bins]
        elevation = output_flip[:, self.num_bins:2*self.num_bins]
        theta = output_flip[:, 2*self.num_bins:3*self.num_bins]
        output_flip = np.concatenate([azimuth[:, ::-1], elevation, theta[:, ::-1]], axis=1).reshape(-1, self.num_bins * 3)

        output = (output + output_flip) / 2.0

        pose_pred = self._prob_to_pose(output)

        pred = {}
        pred['probabilities'] = output
        pred['final'] = [{'azimuth': pose_pred[0, 0], 'elevation': pose_pred[0, 1], 'theta': pose_pred[0, 2]}]

        return pred

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.model.state_dict()
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt
