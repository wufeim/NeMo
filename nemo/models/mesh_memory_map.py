import os

import BboxTools as bbt
import numpy as np

from nemo.utils import cal_occ_one_image
from nemo.utils import CameraTransformer
from nemo.utils import load_off
from nemo.utils import Projector3Dto2D
from nemo.utils.pascal3d_utils import get_anno


def normalization(value):
    return (value - value.min()) / (value.max() - value.min())


def box_include_2d(self_box, other):
    return np.logical_and(
        np.logical_and(
            self_box.bbox[0][0] <= other[:, 0], other[:, 0] < self_box.bbox[0][1]
        ),
        np.logical_and(
            self_box.bbox[1][0] <= other[:, 1], other[:, 1] < self_box.bbox[1][1]
        ),
    )


class MeshLoader:
    def __init__(self, path):
        file_list = os.listdir(path)

        l = len(file_list)
        file_list = ["%02d.off" % (i + 1) for i in range(l)]

        self.mesh_points_3d = []
        self.mesh_triangles = []

        for fname in file_list:
            points_3d, triangles = load_off(os.path.join(path, fname))
            self.mesh_points_3d.append(points_3d)
            self.mesh_triangles.append(triangles)

    def __getitem__(self, item):
        return self.mesh_points_3d[item], self.mesh_triangles[item]

    def __len__(self):
        return len(self.mesh_points_3d)


class MeshConverter:
    def __init__(self, path):
        self.loader = MeshLoader(path=path)

    def get_one(self, annos, return_distance=False):
        off_idx = get_anno(annos, "cad_index")

        points_3d, triangles = self.loader[off_idx - 1]
        points_2d = Projector3Dto2D(annos)(points_3d).astype(np.int32)
        points_2d = np.flip(points_2d, axis=1)
        cam_3d = CameraTransformer(
            annos
        ).get_camera_position()  #  @ np.array([[-1, 0, 0], [0, 0, 1], [0, -1, 0]])

        distance = np.sum((-points_3d - cam_3d.reshape(1, -1)) ** 2, axis=1) ** 0.5
        distance_ = normalization(distance)
        h, w = get_anno(annos, "height", "width")
        map_size = (h, w)

        if_visible = cal_occ_one_image(
            points_2d=points_2d,
            distance=distance_,
            triangles=triangles,
            image_size=map_size,
        )
        box_ori = bbt.from_numpy(get_anno(annos, "box_ori"))
        box_cropped = bbt.from_numpy(get_anno(annos, "box_obj").astype(np.int32))
        box_cropped.set_boundary(
            get_anno(annos, "box_obj").astype(np.int32)[4::].tolist()
        )

        if_visible = np.logical_and(if_visible, box_include_2d(box_ori, points_2d))

        projection_foo = bbt.projection_function_by_boxes(box_ori, box_cropped)

        pixels_2d = projection_foo(points_2d)

        # handle the case that points are out of boundary of the image
        pixels_2d = np.max([np.zeros_like(pixels_2d), pixels_2d], axis=0)
        pixels_2d = np.min(
            [
                np.ones_like(pixels_2d) * (np.array([box_cropped.boundary]) - 1),
                pixels_2d,
            ],
            axis=0,
        )

        if return_distance:
            return pixels_2d, if_visible, distance_

        return pixels_2d, if_visible
