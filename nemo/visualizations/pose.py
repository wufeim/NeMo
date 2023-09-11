import BboxTools as bbt
import numpy as np
from PIL import Image
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, look_at_view_transform, camera_position_from_spherical_angles
from pytorch3d.renderer import MeshRenderer, HardPhongShader, PointLights, TexturesVertex
from pytorch3d.structures import Meshes
import torch

from nemo.utils import rotation_theta, load_off, pre_process_mesh_pascal
from nemo.visualizations.colors import COLORS, COLORSF


def _visualize_pose(pose, mesh_path, img_size, device='cuda:0', color=COLORSF[0]):
    raster_settings = RasterizationSettings(
        image_size=img_size, 
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=0)
    lights = PointLights(device=device, location=((2.0, 2.0, -2.0), ))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=None,
            raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights, cameras=None))
    
    azim = pose['azimuth']
    elev = pose['elevation']
    theta = pose['theta']
    dist = pose['distance']
    px = pose['px']
    py = pose['py']

    if not isinstance(azim, float) and not isinstance(azim, int):
        azim = azim.item()
    if not isinstance(elev, float) and not isinstance(elev, int):
        elev = elev.item()
    if not isinstance(theta, float) and not isinstance(theta, int):
        theta = theta.item()
    if not isinstance(dist, float) and not isinstance(dist, int):
        dist = dist.item()
    if not isinstance(px, float) and not isinstance(px, int):
        px = px.item()
    if not isinstance(py, float) and not isinstance(py, int):
        py = py.item()

    principal = (px, py)

    R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, degrees=False, device=device)
    R = torch.bmm(R, rotation_theta(theta, device_=device))

    this_camera = PerspectiveCameras(focal_length=3000, principal_point=(principal, ), image_size=(img_size, ), in_ndc=False, device=device)

    phong_renderer.rasterizer.cameras = this_camera
    phong_renderer.shader.cameras = this_camera

    verts, faces = pre_process_mesh_pascal(*load_off(mesh_path, to_torch=True))

    mesh = Meshes([verts], [faces]).to(device)
    mesh.textures = TexturesVertex(torch.ones_like(mesh.verts_padded()) * torch.Tensor(color).to(device))

    image_get = phong_renderer(mesh, R=R, T=T)[0].cpu().numpy()

    return image_get


def visualize_pose(pose, mesh_path, img=None, bbox=None, device='cuda:0', alpha=0.6):
    """Visualize an object pose and optionally overlay on the original image.

    Args:
        pose (dict or list(dict)): A dictionary of pose parameters, including azimuth, elevation, theta, distance, px, and py.
        mesh_path (str or list(str)): Path to a CAD mesh (.off).
        img (np.ndarray, optional): Original image. Defaults to None.
        bbox (list or np.ndarray or list(list) or list(np.ndarray), optional): Bounding box of the object in (y0, y1, x0, x1). Defaults to None.
        device (str, optional): Device. Defaults to 'cuda:0'.
        alpha (float, optional): Alpha blending paramter for the rendered object. Defaults to 0.6.

    Returns:
        PIL.Image: Visualization of the pose.
    """
    if img is not None:
        assert isinstance(img, np.ndarray)
        img_size = (img.shape[0], img.shape[1])
    else:
        img_size = (640, 800)

    if isinstance(pose, dict) and isinstance(mesh_path, str):
        pose = [pose]
        mesh_path = [mesh_path]
        bbox = [bbox]

    assert len(pose) == len(mesh_path)
    if bbox is not None:
        assert len(pose) == len(bbox)
    else:
        bbox = [None for p in pose]
    
    img_render = None
    for i, (_pose, _mesh_path) in enumerate(zip(pose, mesh_path)):
        _img = _visualize_pose(_pose, _mesh_path, img_size, device=device, color=COLORSF[i%10])
        _img[..., :3] = _img[..., :3] * 255
        _img = np.clip(np.rint(_img), 0, 255).astype(np.uint8)
        if img_render is None:
            img_render = Image.fromarray(_img)
        else:
            img_render = Image.alpha_composite(img_render, Image.fromarray(_img))
    img_render = np.array(img_render)
    
    if img is not None:
        img = img_render[..., :3] * img_render[..., 3:] * alpha + img * (1 - img_render[..., 3:] * alpha)
    else:
        img = img_render
    
    for i, _bbox in enumerate(bbox):
        if _bbox is not None:
            _bbox = bbt.from_numpy(_bbox)
            img = bbt.draw_bbox(img, _bbox, boundary_width=4, boundary=COLORS[i%10])
    
    img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return Image.fromarray(img)


__all__ = ['visualize_pose']
