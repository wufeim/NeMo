import logging
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


def get_pkg_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, ".."))
    return root


def get_project_root():
    root = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(root, "..", ".."))
    return root


def get_abs_path(path):
    if not os.path.isabs(path):
        path = os.path.join(get_project_root(), path)
    return path


def setup_logging(save_path):
    save_path = get_abs_path(save_path)
    os.makedirs(os.path.join(save_path, "logs"), exist_ok=True)
    os.makedirs(os.path.join(save_path, "ckpts"), exist_ok=True)

    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.root.handlers = []
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=os.path.join(save_path, "logs", f"log_{dt}.txt"),
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)
    return logging.getLogger("").handlers[0].baseFilename


def save_src_files(save_dir, files):
    for f in files:
        if not os.path.isabs(f):
            f = get_abs_path(f)
        shutil.copyfile(f, get_abs_path(os.path.join(save_dir, os.path.basename(f))))


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            import transformers

            transformers.set_seed(seed)
        except ImportError:
            pass
        logging.info(f"Set random seed to {seed}")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def get_param_samples(cfg):
    azimuth_samples = np.linspace(
        cfg.inference.azim_sample.min_pi * np.pi,
        cfg.inference.azim_sample.max_pi * np.pi,
        cfg.inference.azim_sample.num,
        endpoint=False,
    )
    elevation_samples = np.linspace(
        cfg.inference.elev_sample.min_pi * np.pi,
        cfg.inference.elev_sample.max_pi * np.pi,
        cfg.inference.elev_sample.num,
    )
    theta_samples = np.linspace(
        cfg.inference.theta_sample.min_pi * np.pi,
        cfg.inference.theta_sample.max_pi * np.pi,
        cfg.inference.theta_sample.num,
    )
    distance_samples = np.linspace(
        cfg.inference.dist_sample.min,
        cfg.inference.dist_sample.max,
        cfg.inference.dist_sample.num,
    )
    px_samples = np.linspace(
        cfg.inference.px_sample.min,
        cfg.inference.px_sample.max,
        cfg.inference.px_sample.num,
    )
    py_samples = np.linspace(
        cfg.inference.py_sample.min,
        cfg.inference.py_sample.max,
        cfg.inference.py_sample.num,
    )
    return (
        azimuth_samples,
        elevation_samples,
        theta_samples,
        distance_samples,
        px_samples,
        py_samples,
    )
