import math

import numpy as np

CATEGORIES = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
    "chair",
    "diningtable",
    "motorbike",
    "sofa",
    "train",
    "tvmonitor",
]
CATEGORIES_OODCV = [
    "aeroplane",
    "bicycle",
    "boat",
    "bus",
    "car",
    "chair",
    "diningtable",
    "motorbike",
    "sofa",
    "train",
]
KP_LIST = {
    "aeroplane": [
        "left_wing",
        "right_wing",
        "rudder_upper",
        "noselanding",
        "left_elevator",
        "rudder_lower",
        "right_elevator",
        "tail",
    ],
    "bicycle": [
        "seat_front",
        "right_back_wheel",
        "right_pedal_center",
        "right_front_wheel",
        "left_front_wheel",
        "left_handle",
        "seat_back",
        "head_center",
        "left_back_wheel",
        "left_pedal_center",
        "right_handle",
    ],
    "boat": [
        "head_down",
        "head",
        "tail_right",
        "tail_left",
        "head_right",
        "tail",
        "head_left",
    ],
    "bottle": [
        "body",
        "bottom_left",
        "bottom",
        "mouth",
        "body_right",
        "body_left",
        "bottom_right",
    ],
    "bus": [
        "body_front_left_lower",
        "body_front_right_upper",
        "body_back_right_lower",
        "right_back_wheel",
        "body_back_left_upper",
        "right_front_wheel",
        "left_front_wheel",
        "body_front_left_upper",
        "body_back_left_lower",
        "body_back_right_upper",
        "body_front_right_lower",
        "left_back_wheel",
    ],
    "car": [
        "left_front_wheel",
        "left_back_wheel",
        "right_front_wheel",
        "right_back_wheel",
        "upper_left_windshield",
        "upper_right_windshield",
        "upper_left_rearwindow",
        "upper_right_rearwindow",
        "left_front_light",
        "right_front_light",
        "left_back_trunk",
        "right_back_trunk",
    ],
    "chair": [
        "seat_upper_right",
        "back_upper_left",
        "seat_lower_right",
        "leg_lower_left",
        "back_upper_right",
        "leg_upper_right",
        "seat_lower_left",
        "leg_upper_left",
        "seat_upper_left",
        "leg_lower_right",
    ],
    "diningtable": [
        "top_lower_left",
        "top_up",
        "top_lower_right",
        "leg_lower_left",
        "leg_upper_right",
        "top_right",
        "top_left",
        "leg_upper_left",
        "top_upper_left",
        "top_upper_right",
        "top_down",
        "leg_lower_right",
    ],
    "motorbike": [
        "front_seat",
        "right_back_wheel",
        "back_seat",
        "right_front_wheel",
        "left_front_wheel",
        "headlight_center",
        "right_handle_center",
        "left_handle_center",
        "head_center",
        "left_back_wheel",
    ],
    "sofa": [
        "top_right_corner",
        "seat_bottom_right",
        "left_bottom_back",
        "seat_bottom_left",
        "front_bottom_right",
        "top_left_corner",
        "right_bottom_back",
        "seat_top_left",
        "front_bottom_left",
        "seat_top_right",
    ],
    "train": [
        "head_top",
        "mid1_left_bottom",
        "head_left_top",
        "mid1_left_top",
        "mid2_right_bottom",
        "head_right_bottom",
        "mid1_right_bottom",
        "head_left_bottom",
        "mid2_left_top",
        "mid2_left_bottom",
        "head_right_top",
        "tail_right_top",
        "tail_left_top",
        "tail_right_bottom",
        "tail_left_bottom",
        "mid2_right_top",
        "mid1_right_top",
    ],
    "tvmonitor": [
        "back_top_left",
        "back_bottom_right",
        "front_bottom_right",
        "front_top_right",
        "front_top_left",
        "back_bottom_left",
        "back_top_right",
        "front_bottom_left",
    ],
}
IMAGE_SIZES = {
    "aeroplane": (320, 672),
    "bicycle": (608, 608),
    "boat": (384, 672),
    "bottle": (480, 800),
    "bus": (320, 736),
    "chair": (608, 384),
    "car": (256, 672),
    "diningtable": (384, 800),
    "motorbike": (512, 512),
    "sofa": (352, 736),
    "train": (256, 608),
    "tvmonitor": (480, 448),
}
MESH_LEN = {
    "aeroplane": 8,
    "bicycle": 6,
    "boat": 6,
    "bottle": 8,
    "bus": 6,
    "car": 10,
    "chair": 10,
    "diningtable": 6,
    "motorbike": 5,
    "sofa": 6,
    "train": 4,
    "tvmonitor": 4,
}


def get_anno_dict(record, *args, idx=0):
    out = []
    for key_ in args:
        if key_ in ["azimuth", "elevation", "distance", "focal", "theta", "viewport"]:
            out.append(float(record[key_]))
        else:
            out.append(record[key_])

    if len(out) == 1:
        return out[0]

    return tuple(out)


def get_anno(record, *args, idx=0):
    if isinstance(record, dict):
        return get_anno_dict(record, *args, idx=idx)

    out = []
    objects = record["objects"][0]
    viewpoint = record["objects"][0][idx]["viewpoint"][0][0]
    for key_ in args:
        if key_ == "category" or key_ == "class":
            out.append(str(objects[idx]["class"][0]))
        elif key_ == "height":
            out.append(record["imgsize"][0][1])
        elif key_ == "width":
            out.append(record["imgsize"][0][0])
        elif key_ == "bbox":
            out.append(objects[idx]["bbox"][0])
        elif key_ == "cad_index":
            out.append(objects[idx]["cad_index"].item())
        elif key_ == "principal":
            px = viewpoint["px"].item()
            py = viewpoint["py"].item()
            out.append(np.array([px, py]))
        elif key_ in ["theta", "azimuth", "elevation"]:
            if type(viewpoint[key_].item()) == tuple:
                tmp = viewpoint[key_].item()[0]
            else:
                tmp = viewpoint[key_].item()
            out.append(tmp * math.pi / 180)
        elif key_ == "distance":
            if type(viewpoint["distance"].item()) == tuple:
                distance = viewpoint["distance"].item()[0]
            else:
                distance = viewpoint["distance"].item()
            out.append(distance)
        else:
            out.append(viewpoint[key_].item())

    if len(out) == 1:
        return out[0]

    return tuple(out)


def get_obj_ids(record, cate=None):
    check_keys = [
        "azimuth",
        "elevation",
        "distance",
        "focal",
        "theta",
        "principal",
        "viewport",
        "height",
        "width",
        "cad_index",
        "bbox",
    ]
    ids = []
    for i in range(record["objects"][0].shape[0]):
        try:
            get_anno(record, *check_keys, idx=i)
        except IndexError:
            continue
        except ValueError:
            continue

        if cate is not None and get_anno(record, "category", idx=i) != cate:
            continue

        ids.append(i)
    return ids
