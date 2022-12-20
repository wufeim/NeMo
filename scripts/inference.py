import argparse
import logging
import os

import torch
from inference_helpers import helper_func_by_task

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
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
    if cfg.args.cate == 'all':
        all_categories = sorted(list(cfg.dataset.image_sizes.keys()))
    else:
        all_categories = [cfg.args.cate]

    running_results = []
    for cate in all_categories:
        dataset_kwargs = {"data_type": "val", "category": cate}
        val_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs)
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=True, num_workers=1
        )
        logging.info(f"Number of inference images: {len(val_dataset)}")

        model = construct_class_by_name(
            **cfg.model,
            cfg=cfg,
            cate=cate,
            mode="test",
            checkpoint=cfg.args.checkpoint.format(cate),
            device="cuda:0",
        )

        save_pred_path = os.path.join(get_abs_path(cfg.args.save_dir.format(cate)), f'{cfg.dataset.name}_{cate}_val.pth')
        if os.path.isfile(save_pred_path):
            cached_pred = torch.load(save_pred_path)
            results = helper_func_by_task[cfg.task](
                cfg,
                cate,
                model,
                val_dataloader,
                cached_pred=cached_pred
            )
        else:
            results = helper_func_by_task[cfg.task](
                cfg,
                cate,
                model,
                val_dataloader,
            )
            torch.save(results["save_pred"], save_pred_path)

        running_results += results['running']

    helper_func_by_task[cfg.task+'_print'](cfg, all_categories, running_results)


def main():
    args = parse_args()

    setup_logging(args.save_dir.format(args.cate))
    logging.info(args)

    cfg = load_config(args, override=args.opts)

    set_seed(cfg.inference.random_seed)
    save_src_files(args.save_dir.format(args.cate), [args.config, __file__])

    inference(cfg)


if __name__ == "__main__":
    main()
