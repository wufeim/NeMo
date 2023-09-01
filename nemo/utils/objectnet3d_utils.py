import math

import numpy as np
import h5py


CATEGORIES = ['ashtray', 'suitcase', 'clock', 'road_pole', 'bench', 'aeroplane', 'car', 'washing_machine', 
            'jar', 'train', 'bottle', 'pen', 'sign', 'cellphone', 'boat', 'toilet', 'tub', 'refrigerator', 
            'shoe', 'bed', 'bus', 'watch', 'vending_machine', 'laptop', 'headphone', 'skateboard', 
            'screwdriver', 'can', 'lighter', 'computer', 'keyboard', 'desk_lamp', 'stapler', 'shovel', 
            'chair', 'teapot', 'diningtable', 'trophy', 'sofa', 'faucet', 'spoon', 'scissors', 'mouse', 
            'flashlight', 'toaster', 'guitar', 'fire_extinguisher', 'racket', 'printer', 'filing_cabinet', 
            'remote_control', 'door', 'tvmonitor', 'cap', 'bicycle', 'plate', 'microwave', 'cabinet', 'microphone', 
            'fan', 'blackboard', 'paintbrush', 'calculator', 'speaker', 'backpack', 'trash_bin', 'satellite_dish', 
            'stove', 'piano', 'wheelchair', 'slipper', 'toothbrush', 'telephone', 'motorbike', 'pot', 'hammer', 'knife', 
            'rifle', 'helmet', 'skate', 'mailbox', 'eyeglasses', 'camera', 'basket', 'iron', 'dishwasher', 'hair_dryer', 
            'cup', 'kettle', 'bucket', 'comb', 'pencil', 'pan', 'pillow', 'fish_tank', 'coffee_maker', 'key', 'fork', 'eraser', 'bookshelf']


MESH_LEN = { 
 'ashtray': 10,
 'suitcase': 9,
 'clock': 10,
 'road_pole': 16,
 'bench': 7,
 'aeroplane': 8,
 'car': 10,
 'washing_machine': 5,
 'jar': 7,
 'train': 4,
 'bottle': 8,
 'pen': 7,
 'sign': 11,
 'cellphone': 11,
 'boat': 6,
 'toilet': 7,
 'tub': 9,
 'refrigerator': 12,
 'shoe': 10,
 'bed': 10,
 'bus': 6,
 'watch': 9,
 'vending_machine': 11,
 'laptop': 5,
 'headphone': 5,
 'skateboard': 2,
 'screwdriver': 7,
 'can': 6,
 'lighter': 6,
 'computer': 12,
 'keyboard': 12,
 'desk_lamp': 8,
 'stapler': 5,
 'shovel': 5,
 'chair': 10,
 'teapot': 7,
 'diningtable': 6,
 'trophy': 11,
 'sofa': 6,
 'faucet': 14,
 'spoon': 7,
 'scissors': 6,
 'mouse': 5,
 'flashlight': 6,
 'toaster': 10,
 'guitar': 5,
 'fire_extinguisher': 9,
 'racket': 3,
 'printer': 6,
 'filing_cabinet': 8,
 'remote_control': 7,
 'door': 14,
 'tvmonitor': 4,
 'cap': 15,
 'bicycle': 6,
 'plate': 6,
 'microwave': 6,
 'cabinet': 15,
 'microphone': 7,
 'fan': 13,
 'blackboard': 11,
 'paintbrush': 6,
 'calculator': 5,
 'speaker': 9,
 'backpack': 16,
 'trash_bin': 10,
 'satellite_dish': 5,
 'stove': 6,
 'piano': 5,
 'wheelchair': 5,
 'slipper': 6,
 'toothbrush': 5,
 'telephone': 9,
 'motorbike': 5,
 'pot': 7,
 'hammer': 6,
 'knife': 8,
 'rifle': 8,
 'helmet': 8,
 'skate': 2,
 'mailbox': 8,
 'eyeglasses': 11,
 'camera': 11,
 'basket': 15,
 'iron': 5,
 'dishwasher': 4,
 'hair_dryer': 4,
 'cup': 10,
 'kettle': 7,
 'bucket': 4,
 'comb': 9,
 'pencil': 4,
 'pan': 5,
 'pillow': 6,
 'fish_tank': 6,
 'coffee_maker': 7,
 'key': 13,
 'fork': 9,
 'eraser': 15,
 'bookshelf': 8
 }


def get_anno_h5py(record, *args, idx=0):
    out = []
    objects = record["objects"]
    viewpoint = objects[objects['viewpoint'][idx, 0]]
    for key_ in args:
        if key_ == "category" or key_ == "class":
            out.append(''.join([chr(t) for t in np.array(objects[objects['class'][idx, 0]])[:, 0]]))
        elif key_ == "height":
            out.append(int(record['imgsize'][1]))
        elif key_ == "width":
            out.append(int(record['imgsize'][0]))
        elif key_ == "bbox":
            out.append(np.array(objects[np.array(objects['bbox'])[idx, 0]])[:, 0])
        elif key_ == "cad_index":
            out.append(np.array(objects[np.array(objects['cad_index'])[idx, 0]]).item())
        elif key_ == "principal":
            px = np.array(viewpoint["px"]).item()
            py = np.array(viewpoint["py"]).item()
            out.append(np.array([px, py]))
        elif key_ in ["theta", "azimuth", "elevation"]:
            if np.abs(np.array(viewpoint[key_])).sum() == 0 and (key_ + '_coarse') in viewpoint.keys():
                tmp = np.array(viewpoint[key_ + '_coarse']).item()
            else:
                tmp = np.array(viewpoint[key_]).item()
            out.append(tmp * math.pi / 180)
        else:
            out.append(float(viewpoint[key_]))

    if len(out) == 1:
        return out[0]

    return tuple(out)
CATEGORIES = [
    "bed",
    "bookshelf",
    "calculator",
    "cellphone",
    "computer",
    "cabinet",
    "guitar",
    "iron",
    "knife",
    "microwave",
    "pen",
    "pot",
    "rifle",
    "slipper",
    "stove",
    "toilet",
    "tub",
    "wheelchair"
]


MESH_LEN = {
    "bed": 10,
    "bookshelf": 8,
    "calculator": 5,
    "cellphone": 11,
    "computer": 12,
    "cabinet": 15,
    "guitar": 5,
    "iron": 5,
    "knife": 9,
    "microwave": 6,
    "pen": 7,
    "pot": 7,
    "rifle": 8,
    "slipper": 6,
    "stove": 6,
    "toilet": 7,
    "tub": 9,
    "wheelchair": 5
}
