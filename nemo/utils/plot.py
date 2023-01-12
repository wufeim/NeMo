import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns


def plot_score_map(score_map, to_img=True, fname=None, vmin=0.0, vmax=1.0, cbar=False):
    assert to_img ^ (fname is not None)
    ax = sns.heatmap(score_map, square=True, xticklabels=False, yticklabels=False, vmin=vmin, vmax=vmax, cbar=cbar)
    ax.figure.tight_layout()
    if to_img:
        fname = f'tmp_{np.random.randint(0, 100)}.png'
        ax.figure.savefig(fname, bbox_inches='tight', pad_inches=0.0)
        img = np.array(Image.open(fname).convert('RGB'))
        os.system(f'rm {fname}')
        return img
    else:
        ax.figure.savefig(fname, bbox_inches='tight', pad_inches=0.0)
