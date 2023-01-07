import os

import BboxTools as bbt
import cv2
import numpy as np
import scipy.io as sio
from PIL import Image

from nemo.utils import cal_point_weight
from nemo.utils import rle_to_mask
from nemo.utils.pascal3d_utils import get_anno
from nemo.utils.pascal3d_utils import get_obj_ids
from nemo.utils.pascal3d_utils import KP_LIST

mesh_para_names = [
    "azimuth",
    "elevation",
    "theta",
    "distance",
    "focal",
    "principal",
    "viewport",
    "cad_index",
    "bbox",
]


def get_target_distances(start=4.0, end=32.0, num=14):
    ranges = np.linspace(start, end, num + 1)
    return (
        np.random.rand(14).astype(np.float32) * (ranges[1:] - ranges[:-1]) + ranges[:-1]
    )


def prepare_pascal3d_sample(
    cate,
    img_name,
    img_path,
    anno_path,
    occ_level,
    save_image_path,
    save_annotation_path,
    out_shape,
    occ_path=None,
    prepare_mode="first",
    augment_by_dist=False,
    texture_filenames=None,
    texture_path=None,
    single_mesh=True,
    mesh_manager=None,
    direction_dicts=None,
    obj_ids=None,
    extra_anno=None,
    seg_mask_path=None
):
    """
    Prepare a sample for training and validation.

    Parameters
    ----------
    cate: str
    img_name: str
    img_path: str
    anno_path: str
    occ_level: int
    save_image_path: str
    save_annotation_path: str
    out_shape: list
    occ_path: str, default None
    prepare_mode: {'first', 'all'}, default 'first'
    augment_by_dist: bool, default False
    texture_filenames: list, default None
    texture_path: str, default None
    single_mesh: bool, default True
    mesh_manager: MeshConverter, default None
    direction_dicts: dict, default None
    obj_ids: list, default None
    """
    if not os.path.isfile(img_path):
        print(img_path)
        return None
    if not os.path.isfile(anno_path):
        print(anno_path)
        return None

    mat_contents = sio.loadmat(anno_path)
    record = mat_contents["record"][0][0]
    if occ_path is not None:
        occ_mask = np.load(occ_path, allow_pickle=True)["occluder_mask"].astype(np.uint8)
    else:
        occ_mask = None
    if seg_mask_path is not None and os.path.isfile(seg_mask_path):
        rle = np.load(seg_mask_path, allow_pickle=True)
        amodal_mask = rle_to_mask(rle).astype(np.uint8)
    else:
        amodal_mask = None

    if obj_ids is None:
        obj_ids = get_obj_ids(record, cate=cate)
        if len(obj_ids) == 0:
            return None
        if prepare_mode == "first":
            if obj_ids[0] != 0:
                return []
            else:
                obj_ids = [0]

    img = np.array(Image.open(img_path))
    _h, _w = img.shape[0], img.shape[1]

    save_image_names = []
    for obj_id in obj_ids:
        bbox = get_anno(record, "bbox", idx=obj_id)
        box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))

        if get_anno(record, "distance", idx=obj_id) <= 0:
            continue

        if augment_by_dist:
            target_distances = get_target_distances()
        else:
            target_distances = [5.0]

        dist = get_anno(record, "distance", idx=obj_id)
        all_resize_rates = [float(dist / x) for x in target_distances]

        for rr_idx, resize_rate in enumerate(all_resize_rates):
            if resize_rate <= 0.001:
                resize_rate = min(out_shape[0] / box.shape[0], out_shape[1] / box.shape[1])
            try:
                box_ori = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1"))
                box = bbt.from_numpy(bbox, sorts=("x0", "y0", "x1", "y1")) * resize_rate

                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img = np.array(img)
                box_ori = box_ori.set_boundary(img.shape[0:2])

                dsize = (int(img.shape[1] * resize_rate), int(img.shape[0] * resize_rate))
                img = cv2.resize(img, dsize=dsize)
                if occ_mask is not None:
                    occ_mask = cv2.resize(occ_mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)
                if amodal_mask is not None:
                    amodal_mask = cv2.resize(amodal_mask, dsize=dsize, interpolation=cv2.INTER_NEAREST)

                if texture_filenames is not None:
                    texture_name = np.random.choice(texture_filenames)

                center = (
                    get_anno(record, "principal", idx=obj_id)[::-1] * resize_rate
                ).astype(int)
                box1 = bbt.box_by_shape(out_shape, center)
                if (
                    out_shape[0] // 2 - center[0] > 0
                    or out_shape[1] // 2 - center[1] > 0
                    or out_shape[0] // 2 + center[0] - img.shape[0] > 0
                    or out_shape[1] // 2 + center[1] - img.shape[1] > 0
                ):
                    padding = (
                        (
                            max(out_shape[0] // 2 - center[0], 0),
                            max(out_shape[0] // 2 + center[0] - img.shape[0], 0),
                        ),
                        (
                            max(out_shape[1] // 2 - center[1], 0),
                            max(out_shape[1] // 2 + center[1] - img.shape[1], 0),
                        ),
                        (0, 0),
                    )

                    if texture_filenames is None:
                        img = np.pad(img, padding, mode="constant")
                    else:
                        texture_img = Image.open(
                            os.path.join(texture_path, "images", texture_name)
                        )
                        if texture_img.mode != "RGB":
                            texture_img = texture_img.convert("RGB")
                        texture_img = np.array(texture_img)
                        texture_img = cv2.resize(
                            texture_img,
                            dsize=(
                                img.shape[1] + padding[1][0] + padding[1][1],
                                img.shape[0] + padding[0][0] + padding[0][1],
                            ),
                        )
                        texture_img[
                            padding[0][0] : padding[0][0] + img.shape[0],
                            padding[1][0] : padding[1][0] + img.shape[1],
                            :,
                        ] = img
                        img = texture_img

                    if occ_mask is not None:
                        occ_mask = np.pad(occ_mask, (padding[0], padding[1]), mode='constant')
                    if amodal_mask is not None:
                        amodal_mask = np.pad(amodal_mask, (padding[0], padding[1]), mode='constant')

                    box = box.shift([padding[0][0], padding[1][0]])
                    box1 = box1.shift([padding[0][0], padding[1][0]])
                else:
                    padding = ((0, 0), (0, 0), (0, 0))

                box_in_cropped = box.copy()
                box = box1.set_boundary(img.shape[0:2])
                box_in_cropped = box.box_in_box(box_in_cropped)

                bbox = box.bbox
                # img_cropped = box.apply(img)
                img_cropped = img[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], :]
                if occ_mask is not None:
                    occ_mask = occ_mask[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
                if amodal_mask is not None:
                    amodal_mask = amodal_mask[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

                if amodal_mask is not None:
                    if occ_mask is not None:
                        inmodal_mask = amodal_mask * (1 - occ_mask)
                    else:
                        inmodal_mask = amodal_mask
                else:
                    inmodal_mask = None

                """
                proj_foo = bbt.projection_function_by_boxes(
                    box_ori, box_in_cropped, compose=False
                )
                objects = mat_contents["record"]["objects"]
                cropped_kp_list = []
                states_list = []
                for kp in KP_LIST[cate]:
                    states = objects[0, 0]["anchors"][0, 0][kp][0, 0]["status"][0, 0][
                        0, 0
                    ]
                    if states == 1:
                        kp_x, kp_y = objects[0, 0]["anchors"][0, 0][kp][0, 0][
                            "location"
                        ][0, 0][0]
                        if (
                            occ_level > 0
                            and kp_x < occ_mask.shape[1]
                            and kp_y < occ_mask.shape[0]
                            and occ_mask[int(kp_y), int(kp_x)]
                        ):
                            states = 0
                        cropped_kp_x = proj_foo[1](kp_x)
                        cropped_kp_y = proj_foo[0](kp_y)
                    else:
                        cropped_kp_x = cropped_kp_y = 0
                    states_list.append(states)
                    cropped_kp_list.append([cropped_kp_y, cropped_kp_x])
                """
            except KeyboardInterrupt:
                continue

            if augment_by_dist:
                curr_img_name = f"{img_name}_{obj_id:02d}_aug{rr_idx}"
            else:
                curr_img_name = f"{img_name}_{obj_id:02d}"

            save_parameters = dict(
                name=img_name,
                box=box.numpy(),
                box_ori=box_ori.numpy(),
                box_obj=box_in_cropped.numpy(),
                # cropped_kp_list=cropped_kp_list,
                # visible=states_list,
                occ_mask=occ_mask,
                amodal_mask=amodal_mask,
                inmodal_mask=inmodal_mask,
            )
            save_parameters = {
                **save_parameters,
                **{
                    k: v
                    for k, v in zip(
                        mesh_para_names, get_anno(record, *mesh_para_names, idx=obj_id)
                    )
                },
            }
            save_parameters["height"] = _h
            save_parameters["width"] = _w
            save_parameters["resize_rate"] = resize_rate
            save_parameters["padding_params"] = np.array(
                [
                    padding[0][0],
                    padding[0][1],
                    padding[1][0],
                    padding[1][1],
                    padding[2][0],
                    padding[2][1],
                ]
            )

            if texture_filenames is not None:
                save_parameters["texture_name"] = texture_name

            if extra_anno is not None:
                for k in extra_anno:
                    save_parameters[k] = extra_anno[k]

            try:
                # Prepare 3D annotations for NeMo training
                if mesh_manager is not None and direction_dicts is not None:

                    save_parameters["true_cad_index"] = save_parameters["cad_index"]
                    if single_mesh:
                        save_parameters["cad_index"] = 1

                    kps, vis = mesh_manager.get_one(save_parameters)
                    idx = save_parameters["cad_index"] - 1
                    weights = cal_point_weight(
                        direction_dicts[idx],
                        mesh_manager.loader[idx][0],
                        save_parameters,
                    )

                    save_parameters["kp_weights"] = np.abs(weights)
                    save_parameters["cropped_kp_list"] = kps
                    save_parameters["visible"] = vis

                np.savez(
                    os.path.join(save_annotation_path, curr_img_name), **save_parameters
                )
                Image.fromarray(img_cropped).save(
                    os.path.join(save_image_path, curr_img_name + ".JPEG")
                )
                save_image_names.append(
                    (get_anno(record, "cad_index", idx=obj_id), curr_img_name)
                )
            except:
                continue

    return save_image_names
