import sys
sys.path.append('./')

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
# import wandb

from nemo.models.feature_banks import mask_remove_near
from nemo.utils import construct_class_by_name
from nemo.utils import load_config
from nemo.utils import load_off
from nemo.utils import save_src_files
from nemo.utils import set_seed
from nemo.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Training a NeMo model")
    parser.add_argument("--cate", type=str, default="aeroplane")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER, help="Modify config options"
    )
    return parser.parse_args()


def train(cfg):
    dataset_kwargs = {"data_type": "train", "category": cfg.args.cate}
    train_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs)
    if cfg.dataset.sampler is not None:
        train_dataset_sampler = construct_class_by_name(
            **cfg.dataset.sampler, dataset=train_dataset, rank=0, num_replicas=1,
            seed=cfg.training.random_seed)
        shuffle = False
    else:
        train_dataset_sampler = None
        shuffle = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=shuffle,
        num_workers=cfg.training.workers,
        sampler=train_dataset_sampler
    )
    logging.info(f"Number of training images: {len(train_dataset)}")

    # Debug dataset
    if cfg.training.visualize_training_data:
        for i in range(1):
            train_dataset.debug(
                np.random.randint(0, len(train_dataset)), save_dir=cfg.args.save_dir
            )

    if cfg.args.dry_run:
        exit()
    
    model = construct_class_by_name(
        **cfg.model, cfg=cfg, cate=cfg.args.cate, mode='train',
        image_sizes=cfg.dataset.image_sizes)

    logging.info("Start training")
    for epo in range(cfg.training.total_epochs):
        num_iterations = int(cfg.training.scale_iterations_per_epoch * len(train_dataloader))
        for i, sample in enumerate(train_dataloader):

            if i >= num_iterations:
                break
            loss_dict = model.train(sample)
 
            # if i % 300 == 0:
                # logging.info(
                    # f"[Iter {i}] {model.get_training_state()}"
                # )
            # if cfg.use_wandb:
            #     wandb.log(loss_dict)

        if (epo + 1) % cfg.training.log_interval == 0:
            logging.info(
                f"[Epoch {epo+1}/{cfg.training.total_epochs}] {model.get_training_state()}"
            )

        if (epo + 1) % cfg.training.ckpt_interval == 0:
            torch.save(model.get_ckpt(epoch=epo+1, cfg=cfg.asdict()), os.path.join(cfg.args.save_dir, "ckpts", f"model_{epo+1}.pth"))
        model.step_scheduler()


def main():
    args = parse_args()

    setup_logging(args.save_dir)
    logging.info(args)

    cfg = load_config(args, override=args.opts)

    set_seed(cfg.training.random_seed)
    save_src_files(args.save_dir, [args.config, __file__])

    # if cfg.use_wandb:
    #     wandb.init(project=cfg.wandb_project_name, config=cfg.asdict())
    #     wandb.run.name = f'{wandb.run.id}_{os.path.basename(args.save_dir)}'

    train(cfg)


if __name__ == "__main__":
    main()
