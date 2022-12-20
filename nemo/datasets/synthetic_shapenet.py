import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from nemo.utils import get_abs_path


class SyntheticShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, data_camera_mode='shapnet_car', transforms=[], **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        self.root_path = get_abs_path(root_path)
        self.data_camera_mode = data_camera_mode
        self.transforms = transforms

        self.img_path = os.path.join(self.root_path, self.data_type, self.category, 'img')
        self.anno_path = os.path.join(self.root_path, self.data_type, self.category, 'anno')

        self.file_list = [x[:-4] for x in os.listdir(self.img_path) if x.endswith('.png')]

    def __getitem__(self, item):
        name = self.file_list[item]

        ori_img = cv2.imread(os.path.join(self.img_path, f'{name}.png'), cv2.IMREAD_UNCHANGED)
        anno = np.load(os.path.join(self.anno_path, f'{name}.npy'), allow_pickle=True)[()]

        img = ori_img[:, :, :3][:, :, ::-1]
        mask = ori_img[:, :, 3:4]

        condinfo = np.array([
            anno['rotation'],
            np.pi / 2.0 - anno['elevation']
        ], dtype=np.float32)

        sample = {}
        sample['img'] = np.ascontiguousarray(img)
        sample['mask'] = np.ascontiguousarray(mask)
        sample['condinfo'] = condinfo
        return sample

    def __len__(self):
        return len(self.file_list)
