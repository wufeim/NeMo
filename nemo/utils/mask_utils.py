from itertools import groupby

import numpy as np
import pycocotools.mask


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
