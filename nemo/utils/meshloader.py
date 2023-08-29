from nemo.utils import pre_process_mesh_pascal, load_off
import numpy as np

class meshloader():
    def __init__(self, category, base_mesh_path):
        self.all_vertices, self.all_faces = [], []
        self.all_verts_num, self.all_faces_num = [], []
        
        for subcate in category:
            mesh_path_ = base_mesh_path.format(subcate) if "{:s}" in base_mesh_path else base_mesh_path
            mesh_ = pre_process_mesh_pascal(*load_off(mesh_path_, True)) 
            self.all_verts_num.append(mesh_[0].shape[0])
            self.all_faces_num.append(mesh_[1].shape[0]) 
            self.all_vertices.append(mesh_[0].numpy())
            self.all_faces.append(mesh_[1].numpy())
        
        self.max_vert = max(self.all_verts_num)
        self.max_face = max(self.all_faces_num)  

        for i in range(len(category)):
            vert_pad_size = self.max_vert - self.all_verts_num[i]
            face_pad_size = self.max_face - self.all_faces_num[i]
            self.all_vertices[i] = np.pad(self.all_vertices[i], pad_width=((0, vert_pad_size), (0, 0)), mode='constant', constant_values=0)
            self.all_faces[i] = np.pad(self.all_faces[i], pad_width=((0, face_pad_size), (0, 0)), mode='constant', constant_values=-1)
    
    def get_max_vert(self, ):
        return self.max_vert

    def get_verts_num_list(self, ):
        return self.all_verts_num

    def get_mesh_para(self, ):
        return self.all_vertices, self.all_faces