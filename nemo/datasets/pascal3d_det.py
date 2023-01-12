import copy
import os

import BboxTools as bbt
import numpy as np
import torch
import torchvision
from PIL import Image
import skimage
from torch.utils.data import Dataset

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
from nemo.utils import load_off
from nemo.utils.pascal3d_utils import CATEGORIES, CATEGORIES_OODCV


class Pascal3DPlusDet(Dataset):
    def __init__(
        self,
        data_type,
        category,
        root_path,
        transforms,
        mesh_path,
        subtypes=None,
        occ_level=0,
        enable_cache=True,
        weighted=True,
        remove_no_bg=None,
        skip_kp=False,
        segmentation_masks=[],
        **kwargs,
    ):
        self.data_type = data_type
        self.root_path = get_abs_path(root_path)
        self.category = category
        self.subtypes = subtypes if subtypes is not None else {}
        self.occ_level = occ_level
        self.enable_cache = enable_cache
        self.weighted = weighted
        self.remove_no_bg = remove_no_bg
        self.skip_kp = skip_kp
        self.segmentation_masks = segmentation_masks
        self.mesh_path = mesh_path
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )

        if self.category == 'all':
            self.category = CATEGORIES
        elif self.category == 'all_oodcv':
            self.category = CATEGORIES_OODCV
        if not isinstance(self.category, list):
            self.category = [self.category]
        self.multi_cate = len(self.category) > 1

        self.image_path = os.path.join(self.root_path, data_type, "images")
        self.annotation_path = os.path.join(self.root_path, data_type, "annotations")
        self.list_path = os.path.join(self.root_path, data_type, "lists")

        num_verts = []
        for cate in self.category:
            num_verts.append(load_off(os.path.join(self.mesh_path, cate, '01.off'))[0].shape[0])
        self.max_n = max(num_verts)

        file_list = []
        for cate in self.category:
            if self.occ_level == 0:
                _list_path = os.path.join(self.list_path, cate)
            else:
                _list_path = os.path.join(self.list_path, f"{cate}FGL{self.occ_level}_BGL{self.occ_level}")

            if cate not in self.subtypes:
                self.subtypes[cate] = [t.split(".")[0] for t in os.listdir(_list_path)]

            _file_list = sum(
                (
                    [
                        os.path.join(cate if self.occ_level == 0 else f"{cate}FGL{self.occ_level}_BGL{self.occ_level}", l.strip())
                        for l in open(
                            os.path.join(_list_path, subtype_ + ".txt")
                        ).readlines()
                    ]
                    for subtype_ in self.subtypes[cate]
                ),
                [],
            )
            file_list += [(f, cate) for f in _file_list]
        # OOD-CV seems to have duplicate samples -- remove duplicates from file list
        self.file_list = list(set(file_list))
        self.cache = {}

        self.filter()

    def filter(self):
        filtered_file_list = []

        img = self.__getitem__(0)['original_img']
        h, w, _ = img.shape

        for i in range(len(self.file_list)):
            try:
                sample = self.__getitem__(i)
                bbox = sample['boxes'][0]
                if bbox[0] < bbox[2] and bbox[1] < bbox[3]:
                    if bbox[2] > 0 and bbox[3] > 0 and bbox[0] < w and bbox[1] < h:
                        filtered_file_list.append(self.file_list[i])
            except:
                continue
        self.file_list = filtered_file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img, cate = self.file_list[item]

        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
        else:
            img = Image.open(os.path.join(self.image_path, f"{name_img}.JPEG"))
            if img.mode != "RGB":
                img = img.convert("RGB")
            annotation_file = np.load(
                os.path.join(self.annotation_path, name_img.split(".")[0] + ".npz"),
                allow_pickle=True,
            )

            this_name = name_img.split(".")[0]

            boxes = annotation_file['boxes'][:, :4]
            boxes = boxes[:, [2, 0, 3, 1]]
            boxes = torch.as_tensor(boxes, dtype=torch.float32)

            sample = {'img': img}
            if self.transforms:
                sample = self.transforms(sample)

            self.num_bins = 41
            labels = []
            for i in range(len(annotation_file['azimuth'])):
                v = np.array([annotation_file['azimuth'][i], annotation_file['elevation'][i], annotation_file['theta'][i]]) / np.pi
                v[2] += 2 if v[2] < -1 else (-2 if v[2] > 1 else 0)
                vv = v.copy()
                if vv[0] < 0:
                    v[0] = self.num_bins - 1 - np.floor(-vv[0] * self.num_bins / 2.)
                else:
                    v[0] = np.floor(vv[0] * self.num_bins / 2.)
                v[1] = np.ceil(vv[1] * self.num_bins / 2. + self.num_bins / 2. - 1)
                v[2] = np.ceil(vv[2] * self.num_bins / 2. + self.num_bins / 2. - 1)
                v = v.astype(np.int32)
                labels.append(np.array([[1, v[0], v[1], v[2]]]))
            labels = np.concatenate(labels, axis=0)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            sample = {
                "this_name": this_name,
                "img": sample["img"],
                "boxes": boxes[0:1],
                "original_img": np.array(img),
                "azimuth": annotation_file["azimuth"][0],
                "elevation": annotation_file["elevation"][0],
                "theta": annotation_file["theta"][0],
                "distances": torch.as_tensor(np.array([annotation_file['distances']]), dtype=torch.float32)[0:1],
                "labels": labels[0:1]
            }

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        return sample

    def debug(self, item, save_dir=""):
        sample = self.__getitem__(item)
        img = sample["original_img"]

        import cv2
        for b in sample['boxes']:
            x0, y0, x1, y1 = b
            img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        Image.fromarray(img).save(
            os.path.join(save_dir, f'debug_{sample["this_name"].replace("/", "_")}.png')
        )
