import argparse
import multiprocessing
import os
import ssl

import gdown
import numpy as np
import scipy.io as sio
import wget
from create_cuboid_mesh import create_meshes
from tqdm import tqdm

from nemo.models.mesh_memory_map import MeshConverter
from nemo.utils import direction_calculator
from nemo.utils import get_abs_path
from nemo.utils import load_config
from nemo.utils import prepare_pascal3d_sample
from nemo.utils.pascal3d_utils import KP_LIST
from nemo.utils.pascal3d_utils import MESH_LEN


mesh_para_names = [
    "azimuth",
    "elevation",
    "theta",
    "distance",
    "focal",
    "principal",
    "viewport",
    "height",
    "width",
    "cad_index",
    "bbox",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare OOD-CV dataset")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def download_ood_cv(cfg):
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    ood_cv_pose_data_path = get_abs_path(cfg.ood_cv_pose_data_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)

    if os.path.isdir(pascal3d_raw_path):
        print(f"Found Pascal3D+ dataset at {pascal3d_raw_path}")
    else:
        print(f"Downloading Pascal3D+ dataset at {pascal3d_raw_path}")
        wget.download(cfg.pascal3d_raw_url)
        os.system("unzip PASCAL3D+_release1.1.zip")
        os.system("rm PASCAL3D+_release1.1.zip")

    if os.path.isdir(ood_cv_pose_data_path):
        print(f"Found OOD-CV pose dataset at {ood_cv_pose_data_path}")
    else:
        print(f"Downloading OOD-CV pose dataset at {ood_cv_pose_data_path}")
        gdown.download(cfg.ood_cv_pose_url, output="pose.zip", fuzzy=True)
        os.system(f"unzip pose.zip")
        os.system("rm pose.zip")

    if not cfg.pad_texture:
        print("Skipping Describable Textures Dataset")
    elif os.path.isdir(dtd_raw_path):
        print(f"Found Decribable Textures Dataset at {dtd_raw_path}")
    else:
        wget.download(
            "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
        )
        os.system("tar -xf dtd-r1.0.1.tar.gz")
        os.system("rm dtd-r1.0.1.tar.gz")

    mesh_d = "single" if cfg.single_mesh else "multi"
    save_mesh_path = os.path.join(pascal3d_raw_path, f"CAD_{mesh_d}")
    if os.path.isdir(save_mesh_path):
        print(f"Found {mesh_d} meshes at {save_mesh_path}")
    else:
        print(f"Generating {mesh_d} meshes at {save_mesh_path}")
        create_meshes(
            mesh_d,
            os.path.join(pascal3d_raw_path, "CAD"),
            os.path.join(pascal3d_raw_path, f"CAD_{mesh_d}"),
            number_vertices=1000,
            linear_coverage=0.99,
        )


def prepare_ood_cv(cfg, workers=4):
    ood_cv_path = get_abs_path(cfg.root_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    if os.path.isdir(ood_cv_path):
        print(f"Found prepared OOD-CV dataset at {ood_cv_path}")
        return

    if cfg.pad_texture:
        dtd_mat = sio.loadmat(os.path.join(dtd_raw_path, "imdb", "imdb.mat"))
        images = dtd_mat["images"]
        dtd_ids = images[0, 0][0][0]
        dtd_filenames = images[0, 0][1][0]
        dtd_filenames = np.array([f[0] for f in dtd_filenames])
        dtd_splits = images[0, 0][2][0]
        dtd_classes = images[0, 0][3][0]

        train_dtd_filenames = []
        val_dtd_filenames = []
        for j, f, s in zip(dtd_ids, dtd_filenames, dtd_splits):
            if s == 1:
                train_dtd_filenames.append(f)
            elif s == 2:
                val_dtd_filenames.append(f)
        dtd_filenames = {"train": train_dtd_filenames, "val": val_dtd_filenames}

    if cfg.training_only:
        all_set_types = ["train"]
    elif cfg.evaluation_only:
        all_set_types = ["val"]
    else:
        all_set_types = ["train", "val"]

    tasks = []
    for set_type in all_set_types:
        save_root = os.path.join(ood_cv_path, set_type)
        os.makedirs(save_root, exist_ok=True)
        for cate in cfg.image_sizes.keys():
            if cfg.pad_texture:
                tasks.append([cfg, set_type, 0, cate, dtd_filenames[set_type]])
            else:
                tasks.append([cfg, set_type, 0, cate, None])

    with multiprocessing.Pool(workers) as pool:
        results = list(tqdm(pool.imap(worker, tasks), total=len(tasks)))

    total_samples, total_errors = {k: {} for k in all_set_types}, {
        k: {} for k in all_set_types
    }
    for (_err, _total, _set, _cate) in results:
        if _cate not in total_samples[_set]:
            total_samples[_set][_cate] = _total
            total_errors[_set][_cate] = _err
        else:
            total_samples[_set][_cate] += _total
            total_errors[_set][_cate] += _err
    for set_type in all_set_types:
        for cate in cfg.image_sizes.keys():
            print(
                f"Prepared {total_samples[set_type][cate]} {set_type} samples for {cate}, "
                f"error rate {total_errors[set_type][cate]/total_samples[set_type][cate]*100:.2f}%"
            )


def worker(params):
    cfg, set_type, occ, cate, dtd_filenames = params
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    ood_cv_pose_data_path = get_abs_path(cfg.ood_cv_pose_data_path)
    ood_cv_path = get_abs_path(cfg.root_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    save_root = os.path.join(ood_cv_path, set_type)
    occ_mask_dir = None

    this_size = cfg.image_sizes[cate]
    out_shape = [
        ((this_size[0] - 1) // 32 + 1) * 32,
        ((this_size[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]

    if occ == 0:
        data_name = ""
    else:
        data_name = f"FGL{occ}_BGL{occ}"
    save_image_path = os.path.join(save_root, "images", f"{cate}{data_name}")
    save_annotation_path = os.path.join(save_root, "annotations", f"{cate}{data_name}")
    save_list_path = os.path.join(save_root, "lists", f"{cate}{data_name}")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_annotation_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    if set_type == 'train':
        all_pascal3d_samples = ['.'.join(x.split('.')[:-1]) for x in os.listdir(os.path.join(pascal3d_raw_path, 'Images', f'{cate}_imagenet'))]
        all_pascal3d_samples += ['.'.join(x.split('.')[:-1]) for x in os.listdir(os.path.join(pascal3d_raw_path, 'Images', f'{cate}_pascal'))]
        with open(os.path.join(ood_cv_pose_data_path, 'lists', 'TrainSet.txt'), 'r') as fp:
            all_samples = fp.read().strip().split('\n')
        all_samples = [x for x in all_samples if x in all_pascal3d_samples]
    elif set_type == 'val':
        image_names = []
        nuisances = []

        for n in cfg.nuisances:
            if not os.path.isfile(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_{n}.txt')):
                continue
            with open(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_{n}.txt'), 'r') as fp:
                fnames = fp.read().strip().split('\n')
            image_names += fnames
            nuisances += [n] * len(fnames)

        with open(os.path.join(ood_cv_pose_data_path, 'lists', f'{cate}_all.txt'), 'r') as fp:
            fnames = fp.read().strip().split('\n')
        fnames = [x for x in fnames if x not in image_names]
        image_names += fnames
        nuisances += ['iid'] * len(fnames)
        all_samples = [x.split(' ') for x in image_names]
    else:
        raise ValueError(f'Unknown set type: {set_type}')

    manager = MeshConverter(path=os.path.join(get_abs_path(cfg.mesh_path), cate))
    direction_dicts = [direction_calculator(*t) for t in manager.loader]

    num_errors = 0
    mesh_name_list = [[] for _ in range(MESH_LEN[cate])]
    for idx, sample in enumerate(all_samples):
        if set_type == 'train':
            img_name = sample
            if img_name.startswith('n'):
                img_path = os.path.join(pascal3d_raw_path, 'Images', f'{cate}_imagenet', f'{img_name}.JPEG')
                anno_path = os.path.join(pascal3d_raw_path, 'Annotations', f'{cate}_imagenet', f'{img_name}.mat')
            else:
                img_path = os.path.join(pascal3d_raw_path, 'Images', f'{cate}_pascal', f'{img_name}.jpg')
                anno_path = os.path.join(pascal3d_raw_path, 'Annotations', f'{cate}_pascal', f'{img_name}.mat')
            obj_ids = None
        elif set_type == 'val':
            img_name, obj_id = sample
            obj_ids = [int(obj_id)]
            img_path = os.path.join(ood_cv_pose_data_path, 'images', cate, f'{img_name}.JPEG')
            anno_path = os.path.join(ood_cv_pose_data_path, 'annotations', cate, f'{img_name}.mat')
        else:
            raise ValueError(f'Unknown image source: {_src} ({img_name})')

        prepared_sample_names = prepare_pascal3d_sample(
            cate,
            img_name,
            img_path,
            anno_path,
            occ,
            save_image_path=save_image_path,
            save_annotation_path=save_annotation_path,
            out_shape=out_shape,
            occ_path=None
            if occ == 0
            else os.path.join(occ_mask_dir, f"{img_name}.npz"),
            prepare_mode=cfg.prepare_mode,
            augment_by_dist=(set_type == "train" and cfg.augment_by_dist),
            texture_filenames=dtd_filenames,
            texture_path=dtd_raw_path,
            single_mesh=cfg.single_mesh,
            mesh_manager=manager,
            direction_dicts=direction_dicts,
            obj_ids=obj_ids,
            extra_anno=None if set_type == 'train' else {'nuisance': nuisances[idx]},
            center_and_resize=cfg.center_and_resize,
            skip_3d_anno=cfg.skip_3d_anno
        )
        if prepared_sample_names is None:
            num_errors += 1
            continue

        for (cad_index, sample_name) in prepared_sample_names:
            mesh_name_list[cad_index - 1].append(sample_name)

    for i, x in enumerate(mesh_name_list):
        with open(
            os.path.join(save_list_path, "mesh%02d" % (i + 1) + ".txt"), "w"
        ) as fl:
            fl.write("\n".join(x))

    return num_errors, len(all_samples), set_type, cate


def main():
    args = parse_args()
    cfg = load_config(args, load_default_config=False, log_info=False)

    download_ood_cv(cfg)
    prepare_ood_cv(cfg, args.workers)


if __name__ == '__main__':
    main()
