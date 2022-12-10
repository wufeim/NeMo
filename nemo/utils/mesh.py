import BboxTools as bbt
import numpy as np
import pytorch3d
import torch
import torch.nn.functional as F
from pytorch3d.renderer import look_at_rotation
from pytorch3d.renderer.mesh.rasterizer import Fragments


def load_off(off_file_name, to_torch=False):
    file_handle = open(off_file_name)
    # n_points = int(file_handle.readlines(6)[1].split(' ')[0])
    # all_strings = ''.join(list(islice(file_handle, n_points)))

    file_list = file_handle.readlines()
    n_points = int(file_list[1].split(" ")[0])
    all_strings = "".join(file_list[2 : 2 + n_points])
    array_ = np.fromstring(all_strings, dtype=np.float32, sep="\n")

    all_strings = "".join(file_list[2 + n_points :])
    array_int = np.fromstring(all_strings, dtype=np.int32, sep="\n")

    array_ = array_.reshape((-1, 3))

    if not to_torch:
        return array_, array_int.reshape((-1, 4))[:, 1::]
    else:
        return torch.from_numpy(array_), torch.from_numpy(
            array_int.reshape((-1, 4))[:, 1::]
        )


def save_off(off_file_name, vertices, faces):
    out_string = "OFF\n"
    out_string += "%d %d 0\n" % (vertices.shape[0], faces.shape[0])
    for v in vertices:
        out_string += "{:.16f} {:.16f} {:.16f}\n".format(v[0], v[1], v[2])
    for f in faces:
        out_string += "3 %d %d %d\n" % (f[0], f[1], f[2])
    with open(off_file_name, "w") as fl:
        fl.write(out_string)
    return


def camera_position_to_spherical_angle(camera_pose):
    distance_o = torch.sum(camera_pose ** 2, axis=1) ** 0.5
    azimuth_o = torch.atan(camera_pose[:, 0] / camera_pose[:, 2]) % np.pi + np.pi * (
        camera_pose[:, 0] < 0
    ).type(camera_pose.dtype).to(camera_pose.device)
    elevation_o = torch.asin(camera_pose[:, 1] / distance_o)
    return distance_o, elevation_o, azimuth_o


def set_bary_coords_to_nearest(bary_coords_):
    ori_shape = bary_coords_.shape
    exr = bary_coords_ * (bary_coords_ < 0)
    bary_coords_ = bary_coords_.view(-1, bary_coords_.shape[-1])
    arg_max_idx = bary_coords_.argmax(1)
    # return torch.zeros_like(bary_coords_).scatter(1, arg_max_idx.unsqueeze(1), 1.0).view(*ori_shape) + exr
    nearest_target = torch.zeros_like(bary_coords_).scatter(
        1, arg_max_idx.unsqueeze(1), 1.0
    )
    softmax_coords = F.softmax(bary_coords_, dim=1)
    noise = (nearest_target - softmax_coords).detach()
    return (softmax_coords + noise).view(*ori_shape) + exr


def rasterize(R, T, meshes, rasterizer, blur_radius=0):
    # It will automatically update the camera settings -> R, T in rasterizer.camera
    fragments = rasterizer(meshes, R=R, T=T)

    # Copy from pytorch3D source code, try if it is necessary to do gradient decent
    if blur_radius > 0.0:
        clipped_bary_coords = (
            pytorch3d.renderer.mesh.utils._clip_barycentric_coordinates(
                fragments.bary_coords
            )
        )
        clipped_zbuf = pytorch3d.renderer.mesh.utils._interpolate_zbuf(
            fragments.pix_to_face, clipped_bary_coords, meshes
        )
        fragments = Fragments(
            bary_coords=clipped_bary_coords,
            zbuf=clipped_zbuf,
            dists=fragments.dists,
            pix_to_face=fragments.pix_to_face,
        )
    return fragments


# Calculate interpolated maps -> [n, c, h, w]
# face_memory.shape: [n_face, 3, c]
def forward_interpolate(
    R, T, meshes, face_memory, rasterizer, blur_radius=0, mode="bilinear"
):
    fragments = rasterize(R, T, meshes, rasterizer, blur_radius=blur_radius)

    # [n, h, w, 1, d]
    if mode == "nearest":
        out_map = pytorch3d.renderer.mesh.utils.interpolate_face_attributes(
            fragments.pix_to_face,
            set_bary_coords_to_nearest(fragments.bary_coords),
            face_memory,
        )
    else:
        out_map = pytorch3d.renderer.mesh.utils.interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_memory
        )

    if out_map.shape[3] > 1:
        pix_to_face, zbuf, bary_coords, dists = (
            fragments.pix_to_face,
            fragments.zbuf,
            fragments.bary_coords,
            fragments.dists,
        )

        sigma = blur_radius
        gamma = 0.004
        delta = (dists != -1) * 2.0 - 1.0
        D = torch.sigmoid(delta * dists ** 2 / sigma)
        # print('D', torch.min(D), torch.max(D))
        exp_zbuf = torch.exp(zbuf.double() / gamma)
        # print('zbuf', torch.min(zbuf), torch.max(zbuf))
        # print('exp_zbuf', torch.min(exp_zbuf), torch.max(exp_zbuf))
        w = D.double() * exp_zbuf
        # print('w', torch.min(w), torch.max(w))
        w = w / torch.sum(w * (dists != -1), axis=3, keepdim=True)
        w[dists == -1] = 0.0
        # w2 = w1 * (dists != -1)

        # print('w', torch.min(w), torch.max(w))
        # print(torch.mean(w[dists != -1]), torch.mean(w[dists == -1]))
        # print(torch.sum(w))

        d = torch.sum(dists == -1, dim=3)

        # print('out_map', out_map.shape)
        out_map = torch.sum(out_map * w.unsqueeze(4).float(), axis=3)
        # print(out_map.shape)

    out_map = out_map.squeeze(dim=3).transpose(3, 2).transpose(2, 1)
    return out_map


# For meshes in PASCAL3D+
def pre_process_mesh_pascal(verts):
    verts = torch.cat((verts[:, 0:1], verts[:, 2:3], -verts[:, 1:2]), dim=1)
    return verts


def vertex_memory_to_face_memory(memory_bank, faces):
    return memory_bank[faces.type(torch.long)]


def rotation_theta(theta, device_=None):
    # cos -sin  0
    # sin  cos  0
    # 0    0    1
    if type(theta) == float or isinstance(theta, np.floating):
        if device_ is None:
            device_ = "cpu"
        theta = torch.ones((1, 1, 1)).to(device_) * float(theta)
    elif isinstance(theta, np.ndarray):
        theta = torch.ones((1, 1, 1)).to(device_) * float(theta)
    else:
        if device_ is None:
            device_ = theta.device
        theta = theta.view(-1, 1, 1)

    mul_ = (
        torch.Tensor([[1, 0, 0, 0, 1, 0, 0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 0]])
        .view(1, 2, 9)
        .to(device_)
    )
    bia_ = torch.Tensor([0] * 8 + [1]).view(1, 1, 9).to(device_)

    # [n, 1, 2]
    cos_sin = torch.cat((torch.cos(theta), torch.sin(theta)), dim=2).to(device_)

    # [n, 1, 2] @ [1, 2, 9] + [1, 1, 9] => [n, 1, 9] => [n, 3, 3]
    trans = torch.matmul(cos_sin, mul_) + bia_
    trans = trans.view(-1, 3, 3)

    return trans


def campos_to_R_T_det(
    campos, theta, dx, dy, device="cpu", at=((0, 0, 0),), up=((0, 1, 0),)
):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    # translation = translation_matrix(dx, dy, device=device).unsqueeze(0)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    # R = torch.bmm(translation, R)
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]
    # T = T.T.unsqueeze(0)
    # T = torch.bmm(translation, T)[0].T  # (1, 3)
    return R, T


def campos_to_R_T(campos, theta, device="cpu", at=((0, 0, 0),), up=((0, 1, 0),)):
    R = look_at_rotation(campos, at=at, device=device, up=up)  # (n, 3, 3)
    R = torch.bmm(R, rotation_theta(theta, device_=device))
    T = -torch.bmm(R.transpose(1, 2), campos.unsqueeze(2))[:, :, 0]  # (1, 3)
    return R, T


def center_crop_fun(out_shape, max_shape):
    box = bbt.box_by_shape(
        out_shape, (max_shape[0] // 2, max_shape[1] // 2), image_boundary=max_shape
    )
    return lambda x: box.apply(x)
