import copy
import os
import random

import BboxTools as bbt
import numpy as np
import torch
import torchvision
from PIL import Image
import skimage
import skimage.measure
import cv2
from scipy import ndimage
import cv2
from scipy import ndimage
from torch.utils.data import Dataset

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
from nemo.utils import load_off
from nemo.utils.pascal3d_utils import CATEGORIES


class Pascal3DPlus(Dataset):
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
        transforms_test=None,
        segmentation_masks=[],
        training=True,
        **kwargs,
    ):  
        if transforms_test is None:
            transforms_test = transforms
        self.training = training
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
        self.transforms_test = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms_test]
        )
        self.kwargs = kwargs
        self.transforms_test = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms_test]
        )
        self.kwargs = kwargs

        if self.category == 'all':
            self.category = CATEGORIES
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
        if self.remove_no_bg is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                obj_mask = skimage.measure.block_reduce(sample['obj_mask'], (self.remove_no_bg, self.remove_no_bg), np.max)
                if np.sum(1-obj_mask) >= 5:
                    filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list

        if self.segmentation_masks is not None:
            filtered_file_list = []
            for i in range(len(self.file_list)):
                sample = self.__getitem__(i)
                if 'inmodal' in self.segmentation_masks and len(sample['inmodal_mask'].shape) < 2:
                    continue
                if 'amodal' in self.segmentation_masks and len(sample['amodal_mask'].shape) < 2:
                    continue
                filtered_file_list.append(self.file_list[i])
            self.file_list = filtered_file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img, cate = self.file_list[item]
        if '.JPEG' in name_img:
            name_img = name_img.split('.JP')[0]

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

            if "cropped_kp_list" in annotation_file and "visible" in annotation_file:
                kp = annotation_file["cropped_kp_list"]
                iskpvisible = annotation_file["visible"] == 1

                if self.weighted and 'kp_weights' in annotation_file.keys():
                    iskpvisible = iskpvisible * annotation_file["kp_weights"]

                iskpvisible = np.logical_and(
                    iskpvisible, np.all(kp >= np.zeros_like(kp), axis=1)
                )
                iskpvisible = np.logical_and(
                    iskpvisible, np.all(kp < np.array([img.size[::-1]]), axis=1)
                )

                kp = np.max([np.zeros_like(kp), kp], axis=0)
                kp = np.min(
                    [np.ones_like(kp) * (np.array([img.size[::-1]]) - 1), kp], axis=0
                )
            else:
                kp = np.zeros((100, 2), dtype=np.float32)
                iskpvisible = np.zeros((100,), dtype=np.int32)

            this_name = name_img.split(".")[0]

            try:
                box_obj = bbt.from_numpy(annotation_file["box_obj"])
                obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
                # print(box_obj)
                box_obj.assign(obj_mask, 1)
                # print('Not in exception')
            except:
            # except KeyboardInterrupt:
                obj_mask = np.zeros((img.size[1], img.size[0]))

            label = 0 if len(self.category) == 0 else self.category.index(cate)
            
            pad_size = self.max_n - kp.shape[0]
            kp = np.pad(kp, pad_width=((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            iskpvisible = np.pad(iskpvisible, pad_width=(0, pad_size), mode='constant', constant_values=False)
            
            index = np.array([self.max_n * label + k for k in range(self.max_n)])

            sample = {
                "this_name": this_name,
                "cad_index": int(annotation_file["cad_index"]),
                "azimuth": float(annotation_file["azimuth"]) + (0 if self.kwargs.get('add_noise_azimuth', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_azimuth', 0))),
                "elevation": float(annotation_file["elevation"]) + (0 if self.kwargs.get('add_noise_elevation', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_elevation', 0))),
                "theta": float(annotation_file["theta"]) + (0 if self.kwargs.get('add_noise_theta', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_theta', 0))),
                "azimuth": float(annotation_file["azimuth"]) + (0 if self.kwargs.get('add_noise_azimuth', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_azimuth', 0))),
                "elevation": float(annotation_file["elevation"]) + (0 if self.kwargs.get('add_noise_elevation', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_elevation', 0))),
                "theta": float(annotation_file["theta"]) + (0 if self.kwargs.get('add_noise_theta', 0) == 0 else np.random.normal(0, self.kwargs.get('add_noise_theta', 0))),
                "distance": 5,
                "bbox": annotation_file["box_obj"],
                "obj_mask": obj_mask,
                "img": img,
                "original_img": np.array(img),
                "label": label,
                "index": index,
            }
            
            if 'px' in annotation_file.keys():
                sample['principal'] = np.array([annotation_file['px'], annotation_file['py']])
                sample["distance"] = float(annotation_file["distance"])

            if 'amodal' in self.segmentation_masks:
                sample['amodal_mask'] = annotation_file['amodal_mask']
            if 'inmodal' in self.segmentation_masks:
                sample['inmodal_mask'] = annotation_file['inmodal_mask']
            if not self.skip_kp:
                sample['kp'] = kp.astype(np.float32)
                sample['kpvis'] = iskpvisible.astype(bool)

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        if self.training:
            if self.transforms:
                sample = self.transforms(sample)
        
        else:
            if self.transforms_test:
                sample = self.transforms_test(sample)

        return sample

    def debug(self, item, save_dir=""):
        sample = self.__getitem__(item)
        img = sample["original_img"]
        kp, kpvis = sample["kp"], sample["kpvis"]
        y0, y1, x0, x1, _, _ = sample["bbox"]
        obj_mask = sample["obj_mask"]

        import cv2

        for i in range(len(kp)):
            if kpvis[i]:
                img = cv2.circle(
                    img, (int(kp[i, 1]), int(kp[i, 0])), 2, (255, 0, 0), -1
                )
        img = cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

        gray_img = (img * 0.3).astype(np.uint8)
        gray_img[obj_mask == 1] = img[obj_mask == 1]

        Image.fromarray(gray_img).save(
            os.path.join(save_dir, f'debug_{sample["this_name"].replace("/", "_")}.png')
        )


class RandomResize:
    def __init__(self, std=0.3):
        self.std = std

    def __call__(self, sample, resize_rate=None):
        resize_rate = np.random.normal(1, self.std) if resize_rate is None else resize_rate
        if resize_rate < 0.4:
            resize_rate = 0.4
        if resize_rate > 2:
            resize_rate = 2

        assert 'principal' in sample

        img_ = np.array(sample['img'])
        ori_shape = img_.shape[:2]
        out_image = cv2.resize(img_, dsize=(int(resize_rate * ori_shape[1]), int(resize_rate * ori_shape[0])))

        if resize_rate > 1:
            get_image = bbt.box_by_shape(ori_shape[:2], (out_image.shape[0] // 2, out_image.shape[1] // 2), ).apply(out_image)
        else:
            get_image = np.zeros_like(img_)
            bbt.box_by_shape(out_image.shape[:2], (ori_shape[0] // 2, ori_shape[1] // 2), ).assign(get_image, out_image)

        sample['img'] = Image.fromarray(get_image)
        sample['distance'] = sample['distance'] / resize_rate
        sample['principal'] = (sample['principal'] - np.array(ori_shape[::-1]) / 2) * resize_rate + np.array(ori_shape[::-1]) / 2 

        box_obj = sample['bbox']
        for i in range(len(box_obj) - 2):
            if i < 2:
                box_obj[i] = int((box_obj[i] - np.array(ori_shape[::-1])[0] / 2) * resize_rate + np.array(ori_shape[::-1])[0] / 2) 
            else:
                box_obj[i] = int((box_obj[i] - np.array(ori_shape[::-1])[1] / 2) * resize_rate + np.array(ori_shape[::-1])[1] / 2) 

            if box_obj[i] > box_obj[-1]:
                box_obj[i] = box_obj[-1]

            if box_obj[i] < 0:
                box_obj[i] = 0
    
        sample['bbox'] = box_obj
        try:
            box_obj = bbt.from_numpy(box_obj)
            obj_mask = np.zeros(box_obj.boundary, dtype=np.float32)
            box_obj.assign(obj_mask, 1)
            sample['obj_mask'] = obj_mask
        except:
            return sample

        return sample



class RandomRotate:
    def __init__(self, std=20):
        self.std = std

    def __call__(self, sample, rotation_angle=None):
        rotation_angle = np.random.normal(0, self.std) if rotation_angle is None else rotation_angle
        assert 'principal' in sample

        pivot = sample['principal'].astype(np.int32)
        img = np.array(sample['img'])

        padX = [img.shape[1] - pivot[0], pivot[0]]
        padY = [img.shape[0] - pivot[1], pivot[1]]
        imgP = np.pad(img, [padY, padX, [0, 0], ], 'constant')
        
        imgR = ndimage.rotate(imgP, rotation_angle, reshape=False)

        sample['img'] = Image.fromarray(imgR[padY[0] : -padY[1], padX[0] : -padX[1]])
        sample['theta'] += rotation_angle / 180 * np.pi
        
        return sample


class ToTensor:
    def __init__(self):
        self.trans = torchvision.transforms.ToTensor()

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        if "kpvis" in sample and not isinstance(sample["kpvis"], torch.Tensor):
            sample["kpvis"] = torch.Tensor(sample["kpvis"])
        if "kp" in sample and not isinstance(sample["kp"], torch.Tensor):
            sample["kp"] = torch.Tensor(sample["kp"])
        return sample


class Normalize:
    def __init__(self):
        self.trans = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample


def hflip(sample):
    sample["img"] = torchvision.transforms.functional.hflip(sample["img"])
    w = np.array(sample["img"]).shape[1]
    if 'kp' in sample:
        sample["kp"][:, 0] = w - sample["kp"][:, 0] - 1
    sample["principal"][0] = w - sample["principal"][0] - 1
    sample["azimuth"] = np.pi - sample["azimuth"]
    sample["theta"] = - sample["theta"]
    # raise NotImplementedError("Horizontal flip is not tested.")

    return sample


class RandomHorizontalFlip:
    def __init__(self):
        self.trans = torchvision.transforms.RandomApply([lambda x: hflip(x)], p=0.5)

    def __call__(self, sample):
        sample = self.trans(sample)
        return sample


class ColorJitter:
    def __init__(self):
        self.trans = torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.4, hue=0
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample


class Empty:
    def __call__(self, sample):
        return sample

