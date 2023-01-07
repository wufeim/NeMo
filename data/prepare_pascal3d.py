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
from nemo.utils.pascal3d_utils import CATEGORIES
from nemo.utils.pascal3d_utils import MESH_LEN

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
        description="Prepare PASCAL3D+ and Occluded PASCAL3D+"
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


def download_pascal3d(cfg):
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    pascal3d_occ_raw_path = get_abs_path(cfg.pascal3d_occ_raw_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)

    if os.path.isdir(pascal3d_raw_path):
        print(f"Found Pascal3D+ dataset at {pascal3d_raw_path}")
    else:
        print(f"Downloading Pascal3D+ dataset at {pascal3d_raw_path}")
        wget.download(cfg.pascal3d_raw_url)
        os.system("unzip PASCAL3D+_release1.1.zip")
        os.system("rm PASCAL3D+_release1.1.zip")

    if not os.path.isdir(os.path.join(pascal3d_raw_path, "Image_subsets")):
        ssl._create_default_https_context = ssl._create_unverified_context
        wget.download(cfg.image_subsets_url, "Image_subsets.zip")
        os.system("unzip Image_subsets.zip")
        os.system("rm Image_subsets.zip")
        os.system("mv Image_subsets PASCAL3D+_release1.1")

    if max(cfg.occ_levels.train) == 0 and max(cfg.occ_levels.val) == 0:
        print("Skipping OccludedPASCAL3D+")
    elif os.path.isdir(pascal3d_occ_raw_path):
        print(f"Found OccludedPascal3D+ dataset at {pascal3d_occ_raw_path}")
    else:
        os.makedirs(pascal3d_occ_raw_path, exist_ok=True)
        os.chdir(pascal3d_occ_raw_path)
        print(f"Downloading OccludedPascal3D+ dataset at {pascal3d_occ_raw_path}")
        wget.download(cfg.pascal3d_occ_script_url)
        os.system("chmod +x download_FG.sh")
        os.system("sh download_FG.sh")
        os.chdir("..")

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

    if hasattr(cfg, 'segmentation_masks'):
        seg_data_path = get_abs_path(cfg.seg_data_path)
        if os.path.isdir(seg_data_path):
            print(f"Found segmentation data at {seg_data_path}")
        else:
            print(f"Generating segmentation data at {seg_data_path}")
            os.system(f'gdown {cfg.seg_data_url}')
            gdown.download(cfg.seg_data_url, output="Occluded_Vehicles.zip", fuzzy=True)
            os.system('unzip Occluded_Vehicles.zip')
            os.system('rm Occluded_Vehicles.zip')

            # Training
            for cate in CATEGORIES:
                img_path = os.path.join('Occluded_Vehicles', 'training', 'images', f'{cate}_raw')
                anno_path = os.path.join('Occluded_Vehicles', 'training', 'annotations', f'{cate}_raw')

                save_path = os.path.join(seg_data_path, 'train', cate)
                os.makedirs(save_path, exist_ok=True)

                filenames = [x for x in os.listdir(anno_path) if x.endswith('.npz')]
                for fname in tqdm(filenames, desc=f'training_{cate}'):
                    sz = Image.open(os.path.join(img_path, fname.split('.')[0]+'.JPEG')).size
                    annotation = np.load(os.path.join(anno_path, fname), allow_pickle=True)
                    try:
                        mask = pycocotools.mask.decode(pycocotools.mask.merge(pycocotools.mask.frPyObjects(annotation['mask'].tolist(), sz[1], sz[0])))
                        np.save(os.path.join(save_path, fname[:-4]), mask_to_rle(mask))
                    except:
                        continue

            # Validation
            for cate in CATEGORIES:
                for occ_level in [0, 1, 2, 3]:
                    img_path = os.path.join('Occluded_Vehicles', 'testing', 'images', f'{cate}FGL{occ_level}_BGL{occ_level}')
                    anno_path = os.path.join('Occluded_Vehicles', 'testing', 'annotations', f'{cate}FGL{occ_level}_BGL{occ_level}')

                    if occ_level == 0:
                        save_path = os.path.join(seg_data_path, 'val', f'{cate}')
                    else:
                        save_path = os.path.join(seg_data_path, 'val', f'{cate}FGL{occ_level}_BGL{occ_level}')
                    os.makedirs(save_path, exist_ok=True)

                    filenames = [x for x in os.listdir(anno_path) if x.endswith('.npz')]
                    for fname in tqdm(filenames, desc=f'val_{cate}FGL{occ_level}_BGL{occ_level}'):
                        sz = Image.open(os.path.join(img_path, fname.split('.')[0]+'.JPEG')).size
                        annotation = np.load(os.path.join(anno_path, fname))
                        try:
                            mask = pycocotools.mask.decode(pycocotools.mask.merge(pycocotools.mask.frPyObjects(annotation['mask'].tolist(), sz[1], sz[0])))
                            np.save(os.path.join(save_path, fname[:-4]), mask_to_rle(mask))
                        except:
                            continue
    else:
        print("Skipping segmentation data")


def get_target_distances():
    ranges = np.linspace(4.0, 32.0, num=15)
    dists = np.zeros((14,), dtype=np.float32)
    for i in range(14):
        dists[i] = np.random.uniform(ranges[i], ranges[i + 1])
    return dists


def prepare_pascal3d(cfg, workers=4):
    pascal3d_data_path = get_abs_path(cfg.root_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    if os.path.isdir(pascal3d_data_path):
        print(f"Found prepared PASCAL3D+ dataset at {pascal3d_data_path}")
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
    else:
        all_set_types = ["train", "val"]

    tasks = []
    for set_type in all_set_types:
        save_root = os.path.join(pascal3d_data_path, set_type)
        os.makedirs(save_root, exist_ok=True)
        for occ in getattr(cfg.occ_levels, set_type):
            for cate in CATEGORIES:
                if cfg.pad_texture:
                    tasks.append([cfg, set_type, occ, cate, dtd_filenames[set_type]])
                else:
                    tasks.append([cfg, set_type, occ, cate, None])

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
    pascal3d_raw_path = get_abs_path(cfg.pascal3d_raw_path)
    pascal3d_occ_raw_path = get_abs_path(cfg.pascal3d_occ_raw_path)
    dtd_raw_path = get_abs_path(cfg.dtd_raw_path)
    pascal3d_data_path = get_abs_path(cfg.root_path)
    save_root = os.path.join(pascal3d_data_path, set_type)

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

    list_dir = os.path.join(pascal3d_raw_path, "Image_sets")
    pkl_dir = os.path.join(pascal3d_raw_path, "Image_subsets")
    anno_dir = os.path.join(pascal3d_raw_path, "Annotations", f"{cate}_imagenet")
    if occ == 0:
        img_dir = os.path.join(pascal3d_raw_path, "Images", f"{cate}_imagenet")
        occ_mask_dir = ""
    else:
        img_dir = os.path.join(pascal3d_occ_raw_path, "images", f"{cate}{data_name}")
        occ_mask_dir = os.path.join(
            pascal3d_occ_raw_path, "annotations", f"{cate}{data_name}"
        )

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

    download_pascal3d(cfg)
    prepare_pascal3d(cfg, workers=args.workers)


if __name__ == "__main__":
    main()
