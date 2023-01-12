import numpy as np
import torch
import torch.nn as nn
import torchvision

from nemo.models.base_model import BaseModel
from nemo.models.faster_rcnn_helpers import RoIHeadsViewpoint6D
from nemo.utils import construct_class_by_name


class FastRCNNPredictorViewpoint6D(nn.Module):
    def __init__(self, in_channels, num_outputs, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_outputs)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.depth = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        depth = self.depth(x)

        return scores, bbox_deltas, depth


class FasterRCNN(BaseModel):
    def __init__(self, cfg, cate, mode, pretrained=True, num_bins=41, num_classes=12, training=None, checkpoint=None, transforms=[], device='cuda:0', **kwargs):
        super().__init__(cfg, cate, mode, checkpoint, transforms, ['loss', 'loss_cls', 'loss_reg', 'loss_obj', 'loss_rpn', 'loss_dist'], device)
        self.pretrained = pretrained
        self.num_bins = num_bins
        self.num_classes = num_classes
        self.num_outputs = 1 + num_bins * 3 + 1
        self.training_params = training

        self.build()

    def build(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=self.pretrained)

        roi_heads_viewpoint = RoIHeadsViewpoint6D(copy_roi_head=model.roi_heads, num_bins=self.num_bins)
        in_features = roi_heads_viewpoint.box_predictor.cls_score.in_features
        roi_heads_viewpoint.box_predictor = FastRCNNPredictorViewpoint6D(in_features, self.num_outputs, self.num_classes)

        model.roi_heads = roi_heads_viewpoint
        self.model = model.to(self.device)
        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['state'])

        self.optim = construct_class_by_name(params=self.model.parameters(), **self.training_params.optimizer)
        self.scheduler = construct_class_by_name(optimizer=self.optim, **self.training_params.scheduler)

    def train(self, sample):
        self.model.train()
        sample = self.transforms(sample)

        grad_keys = ['boxes', 'labels', 'distances']
        with torch.no_grad():
            images = list(image.to(self.device) for image in sample['img'])
            targets = [
                {k: sample[k][i].to(self.device) for k in grad_keys}
                for i in range(len(sample['img']))
            ]
        loss_dict = self.model(images, targets)
        loss = sum(l for l in loss_dict.values())

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        loss_dict_item = {k: loss_dict[k].item() for k in loss_dict}
        loss_dict_item['loss'] = loss.item()

        self.loss_trackers['loss'].append(loss_dict_item['loss'])
        self.loss_trackers['loss_cls'].append(loss_dict_item['loss_classifier'])
        self.loss_trackers['loss_reg'].append(loss_dict_item['loss_box_reg'])
        self.loss_trackers['loss_obj'].append(loss_dict_item['loss_objectness'])
        self.loss_trackers['loss_rpn'].append(loss_dict_item['loss_rpn_box_reg'])
        self.loss_trackers['loss_dist'].append(loss_dict_item['loss_distance'])

        return loss_dict

    def step_scheduler(self):
        self.scheduler.step()

    def evaluate(self, sample):
        self.model.eval()
        sample = self.transforms(sample)

        with torch.no_grad():
            assert len(sample['img']) == 1
            images = list(image.to(self.device) for image in sample['img'])
            outputs = self.model(images)
            output = [{k: v.detach().cpu().numpy() for k, v in t.items()} for t in outputs][0]

        if output['scores'].shape[0] == 0:
            return {'final': []}
        else:
            pred = {k: v[0] for k, v in output.items()}
            pred['azimuth_bin'] = np.argmax(pred['scores_azimuth'])
            pred['elevation_bin'] = np.argmax(pred['scores_elevation'])
            pred['theta_bin'] = np.argmax(pred['scores_theta'])
            pred['azimuth'] = (pred['azimuth_bin'] + 0.5) * np.pi / (self.num_bins / 2.)
            pred['elevation'] = (pred['elevation_bin'] - self.num_bins / 2) * np.pi / (self.num_bins / 2.)
            pred['theta'] = (pred['theta_bin'] - self.num_bins / 2) * np.pi / (self.num_bins / 2.)
            return {'final': [pred]}

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.model.state_dict()
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt
