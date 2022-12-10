import numpy as np
import torch
import torchvision

from nemo.utils import construct_class_by_name


class BaseModel:
    def __init__(
        self, cfg, cate, mode, checkpoint=None, transforms=[], loss_names=[], device="cuda:0"
    ):
        self.cfg = cfg
        self.cate = cate
        self.mode = mode
        if checkpoint is None:
            assert (
                mode == "train"
            ), "The checkpoint should not be None in validation or test mode"
            self.checkpoint = None
        else:
            self.checkpoint = torch.load(checkpoint, map_location=device)
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )
        self.loss_names = loss_names
        self.device = device

        self.loss_trackers = {}
        for l in loss_names:
            self.loss_trackers[l] = []

    def build(self):
        pass

    def get_training_state(self):
        state_msg = f'lr={self.optim.param_groups[0]["lr"]:.5f}'
        for l in self.loss_names:
            state_msg += f' {l}={np.mean(self.loss_trackers[l])}'
            self.loss_trackers[l] = []
        return state_msg

    def train(self, sample):
        sample = self.transforms(sample)
        raise NotImplementedError

    def evaluate(self, sample):
        sample = self.transforms(sample)
        raise NotImplementedError
