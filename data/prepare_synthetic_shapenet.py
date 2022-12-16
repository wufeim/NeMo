import argparse
import os

from tqdm import tqdm
import wget

from nemo.utils import get_abs_path
from nemo.utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic dataset with ShapeNet models"
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def download_shapenet(cfg):
    shapenet_path = get_abs_path(cfg.shapenet_path)
    if os.path.isdir(shapenet_path):
        print(f"Found ShapeNet at {shapenet_path}")
    else:
        raise FileNotFoundError(f"Cannot find ShapeNet dataset at {shapenet_path}")

    blender_path = get_abs_path(cfg.blender_path)
    if not os.path.isfile(blender_path):
        raise FileNotFoundError(f"Cannot find blender executable at {blender_path}")


def prepare_shapenet(cfg, workers=4):
    root_path = get_abs_path(cfg.root_path)
    shapenet_path = get_abs_path(cfg.shapenet_path)
    os.makedirs(root_path)

    for cate in cfg.synset_name:
        file_list = sorted(os.listdir(os.path.join(shapenet_path, cfg.synset_name[cate])))
        for file in tqdm(file_list):
            os.system(f"{cfg.blender_path} -b -P render_shapenet.py -- --output {root_path} "
                      f"{os.path.join(shapenet_path, cfg.synset_name[cate], file, 'model.obj')} "
                      f"--scale {cfg.render_scale[cate]} --views 24 --resolution 1024 >> tmp.out")


def main():
    args = parse_args()
    cfg = load_config(args, load_default_config=False, log_info=False)

    download_shapenet(cfg)
    prepare_shapenet(cfg, workers=args.workers)


if __name__ == "__main__":
    main()
