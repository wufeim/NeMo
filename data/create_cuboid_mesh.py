import argparse
import os

import numpy as np

from nemo.datasets.pascal3d import CATEGORIES
from nemo.utils import load_off, save_off


def parse_args():
    parser = argparse.ArgumentParser(description='Create cuboid meshes for NeMo')

    parser.add_argument('--CAD_path', type=str, default='PASCAL3D+_release1.1/CAD')
    parser.add_argument('--save_path', type=str, deafult='PASCAL3D+_release1.1/CAD_single')
    parser.add_argument('--mesh_d', type=str, default='single')
    parser.add_argument('--number_vertices', type=int, default=1000)
    parser.add_argument('--linear_coverage', type=float, default=0.99)

    return parser.parse_args()


def meshelize(x_range, y_range, z_range, number_vertices):
    w, h, d = x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]
    total_area = (w * h + h * d + w * d) * 2

    # On average, every vertice attarch 6 edges. Each triangle has 3 edges
    mesh_size = total_area / (number_vertices * 2)

    edge_length = (mesh_size * 2) ** .5

    x_samples = x_range[0] + np.linspace(0, w, int(w / edge_length + 1))
    y_samples = y_range[0] + np.linspace(0, h, int(h / edge_length + 1))
    z_samples = z_range[0] + np.linspace(0, d, int(d / edge_length + 1))

    xn = x_samples.size
    yn = y_samples.size
    zn = z_samples.size

    out_vertices = []
    out_faces = []
    base_idx = 0

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[0]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(yn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[n], z_samples[-1]))
    for m in range(yn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += yn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[0], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(xn):
            out_vertices.append((x_samples[m], y_samples[-1], z_samples[n]))
    for m in range(zn - 1):
        for n in range(xn - 1):
            out_faces.append((base_idx + m * xn + n, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
            out_faces.append((base_idx + (m + 1) * xn + n + 1, base_idx + m * xn + n + 1, base_idx + (m + 1) * xn + n))
    base_idx += zn * xn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[0], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    for n in range(zn):
        for m in range(yn):
            out_vertices.append((x_samples[-1], y_samples[m], z_samples[n]))
    for m in range(zn - 1):
        for n in range(yn - 1):
            out_faces.append((base_idx + m * yn + n, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
            out_faces.append((base_idx + (m + 1) * yn + n + 1, base_idx + m * yn + n + 1, base_idx + (m + 1) * yn + n))
    base_idx += zn * yn

    return np.array(out_vertices), np.array(out_faces)


def create_meshes(mesh_d, CAD_path, save_path, number_vertices, linear_coverage):
    if mesh_d == 'single':
        for cate in CATEGORIES:
            os.makedirs(os.path.join(save_path, cate), exist_ok=True)
            fnames = [x for x in os.listdir(os.path.join(CAD_path, cate)) if x.endswith('.off')]

            vertices = []
            for f in fnames:
                vertices_, faces_ = load_off(os.path.join(CAD_path, cate, f))
                vertices.append(vertices_)

            vertices = np.concatenate(vertices, axis=0)
            selected_shape = int(vertices.shape[0] * linear_coverage)
            out_pos = []

            for i in range(vertices.shape[1]):
                v_sorted = np.sort(vertices[:, i])
                v_group = v_sorted[selected_shape::] - v_sorted[0:-selected_shape]
                min_idx = np.argmin(v_group)
                # print(min_idx, min_idx + selected_shape)
                out_pos.append((v_sorted[min_idx], v_sorted[min_idx + selected_shape]))

            xvert, xface = meshelize(*out_pos, number_vertices=number_vertices)
            save_off(os.path.join(save_path, cate, '01.off'), xvert, xface)
    else:
        raise NotImplementedError


def main():
    args = parse_args()
    create_meshes(args.mesh_d, args.CAD_path, args.save_path,
                  args.number_vertices, args.linear_coverage)


if __name__ == '__main__':
    main()
