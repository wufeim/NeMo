from nemo.utils import pre_process_mesh_pascal, load_off
from pytorch3d.io import load_obj
import numpy as np
import torch


def pascal3D_loader(mesh_path):
    return pre_process_mesh_pascal(*load_off(mesh_path, False)) 


def superclever_loader(mesh_path):
    get = load_obj(mesh_path)
    return get[0].numpy(), get[1].verts_idx.numpy()


class MeshLoader():
    def __init__(self, category, base_mesh_path, loader=pascal3D_loader, pad=True, to_torch=False):
        self.all_vertices, self.all_faces = [], []
        self.all_verts_num, self.all_faces_num = [], []
        
        for subcate in category:
            mesh_path_ = base_mesh_path.format(subcate) if "{:s}" in base_mesh_path else base_mesh_path
            mesh_ = loader(mesh_path_)
            self.all_verts_num.append(mesh_[0].shape[0])
            self.all_faces_num.append(mesh_[1].shape[0]) 
            self.all_vertices.append(mesh_[0])
            self.all_faces.append(mesh_[1])
        
        self.max_vert = max(self.all_verts_num)
        self.max_face = max(self.all_faces_num)  

        if pad:
            for i in range(len(category)):
                vert_pad_size = self.max_vert - self.all_verts_num[i]
                face_pad_size = self.max_face - self.all_faces_num[i]
                self.all_vertices[i] = np.pad(self.all_vertices[i], pad_width=((0, vert_pad_size), (0, 0)), mode='constant', constant_values=0)
                self.all_faces[i] = np.pad(self.all_faces[i], pad_width=((0, face_pad_size), (0, 0)), mode='constant', constant_values=-1)
        
        if to_torch:
            for i in range(len(category)):
                self.all_vertices[i] = torch.from_numpy(self.all_vertices[i])
                self.all_faces[i] = torch.from_numpy(self.all_faces[i])
    
    def get_max_vert(self, ):
        return self.max_vert

    def get_verts_num_list(self, ):
        return self.all_verts_num

    def get_mesh_para(self, ):
        return self.all_vertices, self.all_faces
