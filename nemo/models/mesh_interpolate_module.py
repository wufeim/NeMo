import torch
import torch.nn as nn

try:
    from pytorch3d.structures import Meshes

    use_textures = True
except:
    from pytorch3d.structures import Meshes

    use_textures = False

try:
    from VoGE.Meshes import GaussianMeshesNaive as GaussianMesh
    from VoGE.Converter.Converters import naive_vertices_converter

    enable_voge = True
except:
    enable_voge = False

from pytorch3d.renderer import MeshRasterizer

from nemo.utils import (
    forward_interpolate,
    forward_interpolate_voge,
    pre_process_mesh_pascal,
    vertex_memory_to_face_memory,
    campos_to_R_T,
)


def MeshInterpolateModule(*args, **kwargs):
    rasterizer = kwargs.get('rasterizer')
    if isinstance(rasterizer, MeshRasterizer):
        return MeshInterpolateModuleMesh(*args, **kwargs)
    else:
        assert enable_voge
        return MeshInterpolateModuleVoGE(*args, **kwargs)


class MeshInterpolateModuleVoGE(nn.Module):
    def __init__(self, vertices, faces, memory_bank, rasterizer, post_process=None, off_set_mesh=False, convert_percentage=0.5, **kwargs):
        super(MeshInterpolateModuleVoGE, self).__init__()

        # Convert memory features of vertices to faces
        self.memory = None
        self.update_memory(memory_bank=memory_bank,)

        self.n_mesh = 1
        
        # Preprocess convert meshes in PASCAL3d+ standard to Pytorch3D
        verts = pre_process_mesh_pascal(vertices)

        self.meshes = GaussianMesh(*naive_vertices_converter(verts, faces, percentage=convert_percentage))

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, ):
        self.memory = memory_bank

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super(MeshInterpolateModuleVoGE, self).to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.memory = self.memory.to(device)
        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(self, campos, theta, deform_verts=None, **kwargs):
        R, T = campos_to_R_T(campos, theta, device=campos.device, )

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes
        get = forward_interpolate_voge(R, T, meshes, self.memory.repeat(R.shape[0], 1), rasterizer=self.rasterizer, )

        if self.post_process is not None:
            get = self.post_process(get)
        return get


class MeshInterpolateModuleMesh(nn.Module):
    def __init__(
        self,
        vertices,
        faces,
        memory_bank,
        rasterizer,
        post_process=None,
        off_set_mesh=False,
        **kwargs
    ):
        super().__init__()

        # Convert memory features of vertices to faces
        self.faces = faces
        self.face_memory = None
        self.update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple meshes at same time
        if type(vertices) == list:
            self.n_mesh = len(vertices)
            # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
            verts = [pre_process_mesh_pascal(t) for t in vertices]

            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        else:
            self.n_mesh = 1
            # Preprocess convert meshes in PASCAL3d+ standard to Pytorch3D
            verts = pre_process_mesh_pascal(vertices)

            # Create Pytorch3D meshes
            self.meshes = Meshes(verts=[verts], faces=[faces], textures=None)

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, faces=None):
        if type(memory_bank) == list:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = torch.cat(
                [
                    vertex_memory_to_face_memory(m, f).to(m.device)
                    for m, f in zip(memory_bank, faces)
                ],
                dim=0,
            )
        else:
            if faces is None:
                faces = self.faces
            # Convert memory features of vertices to faces
            self.face_memory = vertex_memory_to_face_memory(memory_bank, faces).to(
                memory_bank.device
            )

    def to(self, *args, **kwargs):
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = args[0]
        super().to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.face_memory = self.face_memory.to(device)
        self.meshes = self.meshes.to(device)
        return self

    def update_rasterizer(self, rasterizer):
        device = self.rasterizer.cameras.device
        self.rasterizer = rasterizer
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(
        self, campos, theta, blur_radius=0, deform_verts=None, mode="bilinear", **kwargs
    ):
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes

        n_cam = campos.shape[0]
        if n_cam > 1 and self.n_mesh > 1:
            get = forward_interpolate(
                R,
                T,
                meshes,
                self.face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )
        elif n_cam > 1 and self.n_mesh == 1:
            get = forward_interpolate(
                R,
                T,
                meshes.extend(campos.shape[0]),
                self.face_memory.repeat(campos.shape[0], 1, 1).view(
                    -1, *self.face_memory.shape[1:]
                ),
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )
        else:
            get = forward_interpolate(
                R,
                T,
                meshes,
                self.face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )

        if self.post_process is not None:
            get = self.post_process(get)
        return get
