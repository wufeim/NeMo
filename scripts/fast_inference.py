import torch
import torch.nn.functional as F

import os
import argparse
import json
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import BboxTools as bbt
from tqdm import tqdm
import sys
import scipy.io
from sklearn.metrics import confusion_matrix
import seaborn as sns
from pylab import savefig

from multiprocessing.pool import ThreadPool

import argparse
import logging
import os
import json

import torch
from inference_helpers import helper_func_by_task

from nemo.utils import construct_class_by_name
from nemo.utils import get_abs_path
from nemo.utils import load_config
from nemo.utils import save_src_files
from nemo.utils import set_seed
from nemo.utils import setup_logging
from VoGE.Utils import Batchifier

parser = argparse.ArgumentParser(description='NeMo Pose Estimation')
parser.add_argument('--type_', default='car', type=str, help='')
parser.add_argument('--mesh_d', default='build', type=str, help='')
parser.add_argument('--turn_off_clutter', action='store_true')
parser.add_argument('--objectnet', default=False, type=bool, help='')
parser.add_argument('--record_pendix', default='', type=str, help='')
parser.add_argument('--pre_render', default=True, type=bool, help='')
parser.add_argument('--data_pendix', default='', type=str, help='')
parser.add_argument('--feature_path', default='saved_features', type=str, help='')
parser.add_argument('--feature_name', default='3D512_points1saved_model_%s_799.pth', type=str, help='')
parser.add_argument('--mesh_path', default='../PASCAL3D/PASCAL3D+_release1.1/CAD_%s/%s/', type=str, help='')
parser.add_argument('--anno_path', default='../data/PASCAL3D_NeMo/annotations/%s/', type=str, help='')

parser.add_argument('--classification', action='store_true', help='')
parser.add_argument('--save_classification', default=None, type=str, help='directory to save classification results')
parser.add_argument('--store_images', action='store_true')
parser.add_argument('--generate_activations', action='store_true')
parser.add_argument('--mask_out_padded', action='store_true')
parser.add_argument('--mask_out_clutter', action='store_true')
parser.add_argument('--set_distance', default=5.0, type=float)

parser.add_argument('--compare_class', default='', type=str)
parser.add_argument('--smaller_fraction', default=None, type=float)
parser.add_argument('--ood_type', default='', type=str)
parser.add_argument('--ood', action='store_true')

parser.add_argument('--save_confusion_matrix_path', default=None, type=str)
parser.add_argument('--visualize_vertex_feat_activation', default=None, type=str)
parser.add_argument('--data_path', default=None, type=str)

parser.add_argument("--cate", type=str, default="aeroplane")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
        "--opts", default=None, nargs=argparse.REMAINDER, help="Modify config options"
    )

# ALL_CLASSES = ["aeroplane", "bicycle", "boat", "bus", "car", "chair", "diningtable", "motorbike", "sofa", "train"]
# ALL_CLASSES = ["aeroplane", "bicycle", "boat", "bus", "car", "bottle", "diningtable", "motorbike", "sofa", "train"]  # exp: 0608
# ALL_CLASSES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "train", "tvmonitor"]  # no Sofa
ALL_CLASSES = ["aeroplane", "bicycle", "boat", "bottle", "bus", "car", "chair", "diningtable", "motorbike", "sofa",
               "train", "tvmonitor"]

args = parser.parse_args()



def normalize(x, dim=0):
    return x / torch.sum(x ** 2, dim=dim, keepdim=True)[0] ** .5


if __name__ == '__main__':
    cfg = load_config(args, override=args.opts)
    set_seed(cfg.inference.random_seed)
    
    
    if cfg.args.cate == 'all':
        all_categories = sorted(list(cfg.dataset.image_sizes.keys()))
    else:
        all_categories = [cfg.args.cate]

    record_file_path_ = None

    device = 'cuda:0'

    counts = {'correct_class': [0] * len(ALL_CLASSES),
              'wrong_class': [0] * len(ALL_CLASSES)}

    sim_scores = {}
    name_list = []
    print(cfg.dataset.occ_level)
    if cfg.inference.classification:
        dataset_kwargs = {"data_type": "val", "category": 'all'}
        val_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs, training=False)
        
    for cate in all_categories:
       

        if not cfg.inference.classification: 
            dataset_kwargs = {"data_type": "val", "category": cate}
            val_dataset = construct_class_by_name(**cfg.dataset, **dataset_kwargs, training=False)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=cfg.inference.get('batch_size', 1), shuffle=False, num_workers=4
        )
        print("Finish setting up validation dataloader")

        model = construct_class_by_name(
            **cfg.model,
            cfg=cfg,
            cate=cate,
            mode="test",
            checkpoint=cfg.args.checkpoint.format(cate),
            device="cuda:0",
        )

        cls_index = ALL_CLASSES.index(cate)

        for i, sample in enumerate(tqdm(val_dataloader)):
            all_similarity = model.fast_inference(sample)
            all_image_name = sample['this_name']
            for j in range(len(all_similarity)):
                similarity_score = all_similarity[j]
                image_name = all_image_name[j]
                if image_name not in sim_scores.keys():
                    sim_scores[image_name] = [0] * len(ALL_CLASSES)
                    name_list.append(image_name)
                    sim_scores[image_name][cls_index] = float(similarity_score)
                else:
                    sim_scores[image_name][cls_index] = float(similarity_score)

    correct = 0
    y_pred = []
    y_true = []
    for image_name in name_list:
        img_cate = image_name.split('/')[0]
        find_F = img_cate.find('F')
        if find_F != -1:
            img_cate = img_cate[:find_F]
        img_class = ALL_CLASSES.index(img_cate)

        y_true.append(img_class)
        y_pred.append(sim_scores[image_name].index(max(sim_scores[image_name])))

        cls_scores = sim_scores[image_name]
        if cls_scores.index(max(cls_scores)) == img_class:
            correct += 1

    accuracy = correct / len(val_dataset)
    print("Accuracy: ", accuracy)

    # Confusion Matrix
    if args.save_confusion_matrix_path:
        cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true, normalize='true').round(2)

        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(cf_matrix, annot=True, fmt='g',
                    ax=ax)  # annot=True to annotate cells, ftm='g' to disable scientific notation

        # labels, title and ticks
        ax.set_xlabel('Predicted labels', fontsize=20)
        ax.set_ylabel('True labels', fontsize=20)
        ax.set_title('Confusion Matrix', fontsize=20)

        ax.xaxis.set_ticklabels(ALL_CLASSES)
        ax.yaxis.set_ticklabels(ALL_CLASSES)

        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        if not os.path.exists(args.save_confusion_matrix_path):
            os.makedirs(args.save_confusion_matrix_path)
        plt.savefig(os.path.join(args.save_confusion_matrix_path, 'occ' + str(cfg.dataset.occ_level) + 'confusion_matrix.jpg'))

    if args.save_classification is not None:
        cls_result_path = os.path.join(args.save_classification, "classification_result")
        if not os.path.exists(cls_result_path):
            os.makedirs(cls_result_path)

        cls_result_path = os.path.join(cls_result_path,
                                       '{}_all_classification_result.json'.format(args.data_pendix))

        with open(cls_result_path, 'w') as fp:
            json.dump(sim_scores, fp)

