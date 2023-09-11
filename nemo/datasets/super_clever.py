import sys
sys.path.append('./')

import numpy as np
import os
import torch
import torchvision
import json
from PIL import Image
from pytorch3d.renderer import (
    MeshRenderer, look_at_view_transform, look_at_rotation,
    RasterizationSettings, MeshRenderer, MeshRasterizer, BlendParams,
    camera_position_from_spherical_angles, HardPhongShader, PointLights,
    PerspectiveCameras, TexturesVertex)
from pytorch3d.transforms import Transform3d, Translate, Scale
from pytorch3d.structures import Meshes, join_meshes_as_scene
from nemo.utils import meshloader
import copy

from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
from nemo.utils import load_off


class SuperClever(Dataset):
    def __init__(
        self,
        root_path,
        transforms,
        mesh_path,
        enable_cache=True,
        transforms_test=None,
        training=True,
        max_objects_in_scenes=10,
        **kwargs,
    ):  
        if transforms_test is None:
            transforms_test = transforms
        self.training = training
        self.root_path = get_abs_path(root_path)
        self.enable_cache = enable_cache
        self.mesh_path = mesh_path
        self.transforms = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms]
        )
        self.transforms_test = torchvision.transforms.Compose(
            [construct_class_by_name(**t) for t in transforms_test]
        )
        self.kwargs = kwargs

        mesh_sub_cates = [t.split('.')[0] for t in os.listdir(mesh_path)]
        self.mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
        self.mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

        self.file_list = ['superCLEVR_new_%06d' % i for i in range(120, 140)]

        self.image_path = os.path.join(self.root_path, "images")
        self.annotation_path = os.path.join(self.root_path, "scenes")
        self.mask_path = os.path.join(self.root_path, "masks")
        self.max_objects_in_scenes = max_objects_in_scenes

        self.cache = {}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        name_img = self.file_list[item]
        
        if self.enable_cache and name_img in self.cache.keys():
            sample = copy.deepcopy(self.cache[name_img])
        else:
            img = Image.open(os.path.join(self.image_path, name_img + '.png'))
            if img.mode != "RGB":
                img = img.convert("RGB")
            anno = json.load(open(os.path.join(self.annotation_path, name_img + '.json')))

            obj_mask = np.array(Image.open(os.path.join(self.mask_path, name_img + '.png'))) / 255
            R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]) 
            
            C = torch.Tensor(anno['camera']['location'])[None] @ R__ 
            R = look_at_rotation(C, )
            T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]
            try:
                sub_cates_indexs = [self.mesh_sub_cates.index(anno['objects'][i]['shape']) for i in range(len(anno['objects']))] + [-1 for _ in range(self.max_objects_in_scenes - len(anno['objects']))]
            except:
                print(name_img)

            label = np.array(sub_cates_indexs)

            objects_transfroms = []

            for i in range(len(anno['objects'])):
                scale_ = anno['objects'][i]['size_num' if 'size_num' in anno['objects'][i].keys() else 'r']
                trans_ = torch.Tensor(anno['objects'][i]['3d_coords'])[None] @ R__

                R_ = look_at_view_transform(dist=1, azim=-anno['objects'][i]['rotation'] + 180)[0][0]
                R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
                
                this_transforms = Transform3d(matrix=R_, ).compose(Scale(scale_, )).compose(Translate(trans_, ))
                objects_transfroms.append(this_transforms.get_matrix()[0])
            objects_transfroms += [torch.zeros((4, 4)) for _ in range(self.max_objects_in_scenes - len(anno['objects']))]

            objects_transfroms = torch.stack(objects_transfroms)

            sample = {
                "this_name": name_img,
                "cad_index": 0,
                "azimuth": 0,
                "elevation": 0,
                "theta": 0,
                "distance": 5,
                "R": R[0],
                "T": T[0],
                "obj_mask": obj_mask,
                "img": img,
                "original_img": np.array(img),
                "label": label,
                "num_objects": len(anno['objects']),
                "transforms": objects_transfroms
            }

            if self.enable_cache:
                self.cache[name_img] = copy.deepcopy(sample)

        if self.training:
            if self.transforms:
                sample = self.transforms(sample)
        
        else:
            if self.transforms_test:
                sample = self.transforms_test(sample)

        return sample

if __name__ == '__main__':
    image_idx = 111
    device = 'cuda'

    home_path = '/home/angtian/data/OmniNeMoSuperClever/data/SuperClever/'

    mesh_path = home_path + 'objs_downsample'
    mesh_sub_cates = [t.split('.')[0] for t in os.listdir(mesh_path)]
    mesh_loader = meshloader.MeshLoader(category=mesh_sub_cates, base_mesh_path=os.path.join(mesh_path, '{:s}.obj'), loader=meshloader.superclever_loader, pad=False, to_torch=True)
    
    mesh_sub_cates = [t.split('_')[1] for t in mesh_sub_cates]

    import tqdm
    for image_idx in tqdm.trange(120, 140):

        anno_ = json.load(open(home_path + "scenes/superCLEVR_new_%06d.json" % image_idx))
        # anno_['objects'] = anno_['objects'][1::]
        img_ = np.array(Image.open(home_path + 'images/superCLEVR_new_%06d.png' % image_idx))
        render_image_size = img_.shape[:2]

        R__ = torch.Tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).to(device) 
        # cameras = PerspectiveCameras(focal_length=700, image_size=(img_.shape[:2], ), principal_point=((img_.shape[1] // 2, img_.shape[0] // 2), ), device=device, in_ndc=False)
        cameras = PerspectiveCameras(focal_length=700 * 2, image_size=(img_.shape[:2], ), principal_point=((img_.shape[1] // 2, img_.shape[0] // 2), ), device=device, in_ndc=False)

        blend_params = BlendParams(sigma=0, gamma=0)
        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        # We can add a point light in front of the object.
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=device, cameras=cameras),
        )

        sub_cates_indexs = [mesh_sub_cates.index(anno_['objects'][i]['shape']) for i in range(len(anno_['objects']))]

        # collected_cads = [mesh_loader.all_vertices[i] for i in sub_cates_indexs]
        objects_transfroms = []

        for i in range(len(sub_cates_indexs)):
            scale_ = anno_['objects'][i]['size_num' if 'size_num' in anno_['objects'][i].keys() else 'r']
            # trans_ = (torch.Tensor(anno_['objects'][i]['3d_coords'])[None] @ R__.cpu()).squeeze()
            trans_ = torch.Tensor(anno_['objects'][i]['3d_coords'])[None] @ R__.cpu()

            # scale_matrix = torch.Tensor([[scale_, 0, 0, trans_[0]], [0, scale_, 0, trans_[1]], [0, 0, scale_, trans_[2]], [0, 0, 0, 1]])

            R_ = look_at_view_transform(dist=1, azim=-anno_['objects'][i]['rotation'] + 180)[0][0]
            R_ = torch.cat([torch.cat([R_, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
            # K_ = scale_matrix # @ R_  # P' = P @ K_
            # objects_transfroms.append(Transform3d(matrix=K_.to(device), device=device))
            objects_transfroms.append(Transform3d(matrix=R_.to(device), device=device).compose(Scale(scale_, device=device)).compose(Translate(trans_.to(device), device=device)))
            
        # rr = Rotation.from_euler('XYZ', anno_['camera']["rotation_euler"], degrees=False).as_matrix()

        C = torch.Tensor(anno_['camera']['location']).to(device)[None] @ R__ 
        R = look_at_rotation(C, device=device)
        T = -torch.bmm(R.transpose(1, 2), C[:, :, None])[:, :, 0]

        verts_ = [objects_transfroms[k].transform_points(mesh_loader.all_vertices[i].to(device)) for k, i in enumerate(sub_cates_indexs)]
        faces_ = [mesh_loader.all_faces[i].to(device) for i in sub_cates_indexs]
        textures = TexturesVertex([torch.ones_like(vv) for vv in verts_])
        mesh_ = join_meshes_as_scene(Meshes(verts_, faces_, textures=textures))
        # import ipdb
        # ipdb.set_trace()

        img = phong_renderer(mesh_, R=R, T=T).cpu().numpy()[0]

        rate_ = 0.8
        img_out = img[..., :3] * img[..., 3:] * rate_ * 255 + img_[..., :3] * (1 - img[..., 3:] * rate_)
        # Image.fromarray(img_out.astype(np.uint8)).save('debug/super_clever_cc2.png')
        Image.fromarray((img[..., 3] * 255).astype(np.uint8)).save(home_path + 'masks/superCLEVR_new_%06d.png' % image_idx)








        # import ipdb
        # ipdb.set_trace()






