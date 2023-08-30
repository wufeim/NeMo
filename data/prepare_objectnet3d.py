import argparse
import multiprocessing
import os
import ssl

import numpy as np
import scipy.io as sio
import wget
import pycocotools.mask
from tqdm import tqdm

from nemo.models.mesh_memory_map import MeshConverter
from nemo.utils import direction_calculator
from nemo.utils import get_abs_path
from nemo.utils import load_config
from nemo.utils import prepare_pascal3d_sample
from nemo.utils.objectnet3d_utils import CATEGORIES
from nemo.utils.objectnet3d_utils import MESH_LEN

from create_cuboid_mesh import create_meshes

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
    parser = argparse.ArgumentParser(
        description="Prepare ObjectNet3D"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def mask_to_rle(mask):
    mask = np.asfortranarray(mask)
    rle = {'counts': [], 'size': list(mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle_to_mask(rle):
    if isinstance(rle, np.ndarray):
        rle = rle[()]
    compressed_rle = pycocotools.mask.frPyObjects(rle, rle.get('size')[0], rle.get('size')[1])
    return pycocotools.mask.decode(compressed_rle).astype(np.uint8)


def download_objectnet3d(cfg):
    objectnet3d_raw_path = get_abs_path(cfg.objectnet3d_raw_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)

    if os.path.isdir(objectnet3d_raw_path):
        print(f"Found ObjectNet3D dataset at {objectnet3d_raw_path}")
    else:
        print(f"Downloading ObjectNet3D dataset at {objectnet3d_raw_path}")
        wget.download(cfg.objectnet3d_img_url)
        wget.download(cfg.objectnet3d_cad_url)
        wget.download(cfg.objectnet3d_anno_url)
        wget.download(cfg.objectnet3d_set_url)
        wget.download(cfg.objectnet3d_lists_url)
        os.system("unzip ObjectNet3D_annotations.zip")
        os.system("unzip ObjectNet3D_cads.zip")
        os.system("unzip ObjectNet3D_image_sets.zip")
        os.system("unzip ObjectNet3D_images.zip")
        os.system("unzip ObjectNet3D_lists.zip")
        os.system("rm ObjectNet3D_annotations.zip")
        os.system("rm ObjectNet3D_cads.zip")
        os.system("rm ObjectNet3D_image_sets.zip")
        os.system("rm ObjectNet3D_images.zip")
        os.system("rm ObjectNet3D_lists.zip")
        if os.path.join(os.getcwd(), 'ObjectNet3D') != objectnet3d_raw_path:
            os.system(f"mv ObjectNet3D {objectnet3d_raw_path}")

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
    save_mesh_path = os.path.join(objectnet3d_raw_path, f"CAD_{mesh_d}")
    if os.path.isdir(save_mesh_path):
        print(f"Found {mesh_d} meshes at {save_mesh_path}")
    else:
        print(f"Generating {mesh_d} meshes at {save_mesh_path}")
        create_meshes(
            mesh_d,
            os.path.join(objectnet3d_raw_path, "CAD", "off"),
            os.path.join(objectnet3d_raw_path, f"CAD_{mesh_d}"),
            number_vertices=1000,
            linear_coverage=0.99,
            categories=CATEGORIES
        )


def get_target_distances():
    ranges = np.linspace(4.0, 32.0, num=15)
    dists = np.zeros((14,), dtype=np.float32)
    for i in range(14):
        dists[i] = np.random.uniform(ranges[i], ranges[i + 1])
    return dists


def prepare_objectnet3d(cfg, workers=4):
    objectnet3d_data_path = get_abs_path(cfg.root_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    if os.path.isdir(objectnet3d_data_path):
        print(f"Found prepared ObjectNet3D dataset at {objectnet3d_data_path}")
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
        save_root = os.path.join(objectnet3d_data_path, set_type)
        os.makedirs(save_root, exist_ok=True)
        for cate in CATEGORIES:
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
        for cate in CATEGORIES:
            print(
                f"Prepared {total_samples[set_type][cate]} {set_type} samples for {cate}, "
                f"error rate {total_errors[set_type][cate]/total_samples[set_type][cate]*100:.2f}%"
            )


def worker(params):
    cfg, set_type, occ, cate, dtd_filenames = params
    objectnet3d_raw_path = get_abs_path(cfg.objectnet3d_raw_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    objectnet3d_data_path = get_abs_path(cfg.root_path)
    save_root = os.path.join(objectnet3d_data_path, set_type)

    this_size = cfg.image_sizes[cate]
    out_shape = [
        ((this_size[0] - 1) // 32 + 1) * 32,
        ((this_size[1] - 1) // 32 + 1) * 32,
    ]
    out_shape = [int(out_shape[0]), int(out_shape[1])]

    data_name = ""
    save_image_path = os.path.join(save_root, "images", f"{cate}{data_name}")
    save_annotation_path = os.path.join(save_root, "annotations", f"{cate}{data_name}")
    save_list_path = os.path.join(save_root, "lists", f"{cate}{data_name}")
    os.makedirs(save_image_path, exist_ok=True)
    os.makedirs(save_annotation_path, exist_ok=True)
    os.makedirs(save_list_path, exist_ok=True)

    list_dir = os.path.join(objectnet3d_raw_path, "lists")
    anno_dir = os.path.join(objectnet3d_raw_path, "Annotations")
    img_dir = os.path.join(objectnet3d_raw_path, "Images")

    list_fname = os.path.join(list_dir, f"{cate}_imagenet_{set_type}.txt")
    with open(list_fname) as fp:
        image_names = fp.readlines()
    image_names = [x.strip() for x in image_names if x != "\n"]

    if cfg.mesh_path is not None:
        manager = MeshConverter(path=os.path.join(get_abs_path(cfg.mesh_path), cate))
        direction_dicts = [direction_calculator(*t) for t in manager.loader]
    else:
        manager = direction_dicts = None

    num_errors = 0
    mesh_name_list = [[] for _ in range(MESH_LEN[cate])]
    for img_name in image_names:
        img_path = os.path.join(img_dir, f"{img_name}.JPEG")
        anno_path = os.path.join(anno_dir, f"{img_name}.mat")

        prepared_sample_names = prepare_pascal3d_sample(
            cate,
            img_name,
            img_path,
            anno_path,
            occ,
            save_image_path=save_image_path,
            save_annotation_path=save_annotation_path,
            out_shape=out_shape,
            occ_path=None,
            prepare_mode=cfg.prepare_mode,
            augment_by_dist=(set_type == "train" and cfg.augment_by_dist),
            texture_filenames=dtd_filenames,
            texture_path=dtd_raw_path,
            single_mesh=cfg.single_mesh,
            mesh_manager=manager,
            direction_dicts=direction_dicts,
            seg_mask_path=None,
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

    return num_errors, len(image_names), set_type, cate


def main():
    args = parse_args()
    cfg = load_config(args, load_default_config=False, log_info=False)

    download_objectnet3d(cfg)
    prepare_objectnet3d(cfg, workers=args.workers)


if __name__ == "__main__":
    main()
