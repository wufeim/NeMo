import math

import numpy as np

from nemo.utils.pascal3d_utils import get_anno


class CameraTransformer:
    parameters_3d_to_2d = (
        "azimuth, elevation, distance, focal, theta, principal, viewport"
    )
    parameters_2d_to_3d = "theta, focal, principal, viewport"
    parameters_transformation_matrix = "azimuth, elevation, distance"
    parameters_camera_polygon = "height, width, theta, focal, principal, viewport"

    def __init__(self, record):
        self.record = record

    def project_points_3d_to_2d(self, x):
        return project_points_3d_to_2d(
            x, *get_anno(self.record, *self.parameters_3d_to_2d.split(", "))
        )

    def project_points_2d_to_3d(self, x):
        return project_points_2d_to_3d(
            x, *get_anno(self.record, *self.parameters_2d_to_3d.split(", "))
        )

    def get_camera_polygon(self):
        return get_camera_polygon(
            *get_anno(self.record, *self.parameters_camera_polygon.split(", "))
        )

    def get_transformation_matrix(self):
        return get_transformation_matrix(
            *get_anno(self.record, *self.parameters_transformation_matrix.split(", "))
        )

    def get_camera_position(self):
        return get_camera_position(self.get_transformation_matrix())


class Projector3Dto2D(CameraTransformer):
    def __call__(self, x):
        return self.project_points_3d_to_2d(x)


class Projector2Dto3D(CameraTransformer):
    def __call__(self, x):
        return self.project_points_2d_to_3d(x)


def project_points_3d_to_2d(
    x3d,
    azimuth,
    elevation,
    distance,
    focal,
    theta,
    principal,
    viewport,
):
    R = get_transformation_matrix(azimuth, elevation, distance)
    if R is None:
        return np.empty(0)

    # perspective project matrix
    # however, we set the viewport to 3000, which makes the camera similar to
    # an affine-camera.
    # Exploring a real perspective camera can be a future work.
    M = viewport
    P = np.array([[M * focal, 0, 0], [0, M * focal, 0], [0, 0, -1]]).dot(R[:3, :4])

    # project
    x3d_ = np.hstack((x3d, np.ones((len(x3d), 1)))).T
    x2d = np.dot(P, x3d_)
    x2d[0, :] = x2d[0, :] / x2d[2, :]
    x2d[1, :] = x2d[1, :] / x2d[2, :]
    x2d = x2d[0:2, :]

    # rotation matrix 2D
    R2d = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    x2d = np.dot(R2d, x2d).T

    # transform to image coordinate
    x2d[:, 1] *= -1
    x2d = x2d + np.repeat(principal[np.newaxis, :], len(x2d), axis=0)

    return x2d


def get_transformation_matrix(azimuth, elevation, distance):
    if distance == 0:
        # return None
        distance = 0.1

    # camera center
    C = np.zeros((3, 1))
    C[0] = distance * math.cos(elevation) * math.sin(azimuth)
    C[1] = -distance * math.cos(elevation) * math.cos(azimuth)
    C[2] = distance * math.sin(elevation)

    # rotate coordinate system by theta is equal to rotating the model by theta
    azimuth = -azimuth
    elevation = -(math.pi / 2 - elevation)

    # rotation matrix
    Rz = np.array(
        [
            [math.cos(azimuth), -math.sin(azimuth), 0],
            [math.sin(azimuth), math.cos(azimuth), 0],
            [0, 0, 1],
        ]
    )  # rotation by azimuth
    Rx = np.array(
        [
            [1, 0, 0],
            [0, math.cos(elevation), -math.sin(elevation)],
            [0, math.sin(elevation), math.cos(elevation)],
        ]
    )  # rotation by elevation
    R_rot = np.dot(Rx, Rz)
    R = np.hstack((R_rot, np.dot(-R_rot, C)))
    R = np.vstack((R, [0, 0, 0, 1]))

    return R


def project_points_2d_to_3d(x2d, theta, focal, principal, viewport):
    x2d = x2d.copy()
    # rotate the camera model
    R2d = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
    )
    # projection matrix
    M = viewport
    P = np.array(
        [
            [M * focal, 0, 0],
            [0, M * focal, 0],
            [0, 0, -1],
        ]
    )
    x2d -= principal
    x2d[:, 1] *= -1
    x2d = np.dot(np.linalg.inv(R2d), x2d.T).T
    x2d = np.hstack((x2d, np.ones((len(x2d), 1), dtype=np.float64)))
    x2d = np.dot(np.linalg.inv(P), x2d.T).T
    return x2d


def get_camera_polygon(height, width, theta, focal, principal, viewport):
    x0 = np.array([0, 0, 0], dtype=np.float64)

    # project the 3D points
    x = np.array(
        [
            [0, 0],
            [width, 0],
            [width, height],
            [0, height],
        ],
        dtype=np.float64,
    )
    x = project_points_2d_to_3d(x, theta, focal, principal, viewport)

    x = np.vstack((x0, x))

    return x


def get_camera_position(projection_matrix):
    pro_ = projection_matrix[0:3, 0:3]
    pro_ = np.linalg.pinv(pro_)

    f_ = projection_matrix[0:3, 3:]

    return np.matmul(pro_, f_).ravel()
