import numpy as np
from tqdm import tqdm

from nemo.utils import pose_error


def inference_3d_pose_estimation(
    cfg,
    model,
    dataloader,
):
    pose_errors = []
    for i, sample in enumerate(tqdm(dataloader, desc=f"{cfg.task}_{cfg.args.cate}")):
        pred = model.predict(sample)
        pose_errors.append(pose_error(sample, pred["final"][0]))
    pose_errors = np.array(pose_errors)

    results = {}
    results["pose_errors"] = pose_errors
    results["pi6_acc"] = np.mean(pose_errors < np.pi / 6)
    results["pi18_acc"] = np.mean(pose_errors < np.pi / 18)
    results["med_err"] = np.median(pose_errors) / np.pi * 180.0

    return results


helper_func_by_task = {"3d_pose_estimation": inference_3d_pose_estimation}
