import argparse
import logging

import torch
from inference_helpers import helper_func_by_task

from nemo.utils import construct_class_by_name
from nemo.utils import load_config
from nemo.utils import save_src_files
from nemo.utils import set_seed
from nemo.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Training a NeMo model")
    parser.add_argument("--cate", type=str, default="aeroplane")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER, help="Modify config options"
    )
    return parser.parse_args()


def inference(cfg):
    dataset_kwargs = {"data_type": "val", "category": cfg.args.cate}
    val_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=1
    )
    logging.info(f"Number of inference images: {len(val_dataset)}")

    model = construct_class_by_name(
        **cfg.model,
        cfg=cfg,
        cate=cfg.args.cate,
        mode="test",
        checkpoint=cfg.args.checkpoint,
        device="cuda:0",
    )

    results = helper_func_by_task[cfg.task](
        cfg,
        model,
        val_dataloader,
    )

    if cfg.task == "3d_pose_estimation":
        print(f"\n3D Pose Estimation Results:")
        print(f"Dataset:     {cfg.dataset.name} (num={len(val_dataset)})")
        print(f"Model:       {cfg.model.name} (ckpt={cfg.args.checkpoint})")
        print(f'pi/6 acc:    {results["pi6_acc"]*100:.2f}%')
        print(f'pi/18 acc:   {results["pi18_acc"]*100:.2f}%')
        print(f'Median err:  {results["med_err"]:.2f}')
    else:
        raise NotImplementedError


def main():
    args = parse_args()

    setup_logging(args.save_dir)
    logging.info(args)

    cfg = load_config(args, override=args.opts)

    set_seed(cfg.inference.random_seed)
    save_src_files(args.save_dir, [args.config, __file__])

    inference(cfg)


if __name__ == "__main__":
    main()
