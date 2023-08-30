import argparse
import json
import os
from sklearn.metrics import confusion_matrix
import numpy as np

CATEGORIES = [
    "aeroplane",
    "bicycle",
    "boat",
    "bottle",
    "bus",
    "car",
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

def parse_args():
    parser = argparse.ArgumentParser(description="Organizing Classification Accuracy")
    parser.add_argument("--save_cls_dir", type=str, required=True)
    parser.add_argument("--ood_cv", action='store_true')
    parser.add_arugment("--occ_level", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.ood_cv:
        category_list = CATEGORIES_OODCV
    else:
        category_list = CATEGORIES
    json_dirs = []
    organize_cls = {}
    cate_indexes = []
    for cate in category_list:
        json_dirs.append(os.path.join(args.save_cls_dir, "pascal3d_occ{}_{}_cls_val.json".format(args.occ_level, cate)))
        cate_indexes.append(category_list.index(cate))

    for json_dir in json_dirs:
        count = 0
        with open(json_dir) as f:
            current_cls_result = json.load(f)
            cate_index = cate_indexes[json_dirs.index(json_dir)]
            for img_name in current_cls_result.keys():

                cls_score, pose_score = current_cls_result[img_name]
                
                img_cate = img_name.split('/')[0]
                # HANDLE CASE IF IMAGENAME INCLUDE FG_BG NAME
                find_F = img_cate.find('F')
                if find_F != -1:
                    img_cate = img_cate[:find_F]
                
                img_label = category_list.index(img_cate)

                if img_name not in organize_cls.keys():
                    organize_cls[img_name] = ([100 for i in range(len(category_list))], img_label, [100 for i in range(len(category_list))])
                organize_cls[img_name][0][cate_index] = cls_score
                organize_cls[img_name][2][cate_index] = pose_score
                
    y_true, y_pred, y_error = [], [], []
    error_dict = {}
    count = 0
    for cate in category_list:
        error_dict[cate] = []
    for img_name in organize_cls.keys():
        
        cls_score, true_label, pose_score = organize_cls[img_name]
        pred_label = cls_score.index(min(cls_score))

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_error.append(pose_score[true_label])
        error_dict[category_list[true_label]].append(pose_score[true_label])
    
    
    cf_matrix = confusion_matrix(y_pred=y_pred, y_true=y_true)
    print(cf_matrix)

    total_image_count = len(y_true)
    ground_truth_count = list(0 for i in range(len(category_list)))
    pred_count = list(0 for i in range(len(category_list)))
    pi_over_6 = list(0 for i in range(len(category_list)))
    pi_over_18 = list(0 for i in range(len(category_list)))
    for i in range(len(y_true)):
        label = y_true[i]
        ground_truth_count[label] += 1
        if y_pred[i] == label:
            pred_count[label] += 1
            current_error = y_error[i]
            if current_error <= np.pi / 6:
                pi_over_6[label] += 1
            if current_error <= np.pi / 18:
                pi_over_18[label] += 1

    
    for i in range(len(category_list)):
        if ground_truth_count[i] == 0:
            print('The accuaracy of class ' + category_list[i] + ' is ' + str(0)) 
        else:
            acc = pred_count[i] / ground_truth_count[i]
            print('The accuaracy of class ' + category_list[i] + ' is ' + str(acc))
        


    acc = sum(pred_count) / total_image_count
    print('The aggregated accuaracy is ' + str(acc))

    error1 = (sum(pi_over_6)) / (total_image_count)
    print('Aggregated portion of images rightly classified with error under pi/6 is ' + str(error1))

    error2 = (sum(pi_over_18)) / (total_image_count)
    print('Aggregated portion of images rightly classified with error under pi/18 is ' + str(error2))



    if args.ood_cv:
        cates1 = ['plane', 'bike', 'boat', 'bus', 'car', 'chair', 'table', 'mbike', 'sofa', 'train']
    else:
        cates1 = ['plane', 'bike', 'boat', 'bottle', 'bus', 'car', 'chair', 'table', 'mbike', 'sofa', 'train', 'tv']

    for name in error_dict.keys():
        total_error = np.array(error_dict[name])

    out_ = np.zeros((4, len(category_list)), dtype=np.float32)
    i = 0
    for name in error_dict.keys():
        total_error = np.array(error_dict[name])
        if len(total_error) == 0:
            continue
        out_[0, i] = float(np.mean(np.array(total_error) < np.pi / 6)) * 100
        out_[1, i] = float(np.mean(np.array(total_error) < np.pi / 18)) * 100
        out_[2, i] = float(180 / np.pi * np.median(np.array(total_error)))
        out_[3, i] = np.array(total_error).size
        i += 1

    
    print('Metric | ', end='')
    for cate in cates1:
        print(cate, end='\t')
    print('Mean')

    print('Pi/6   | ', end='')
    for i, _ in enumerate(category_list):
        print('%.1f' % out_[0, i], end='\t')
    print('%.1f' % (np.sum(out_[3, :] * out_[0, :]) / np.sum(out_[3, :])))

    print('Pi/18  | ', end='')
    for i, _ in enumerate(category_list):
        print('%.1f' % out_[1, i], end='\t')
    print('%.1f' % (np.sum(out_[3, :] * out_[1, :]) / np.sum(out_[3, :])))

    print('MedErr | ', end='')
    for i, _ in enumerate(category_list):
        print('%.1f' % out_[2, i], end='\t')
    print('%.1f' % (np.sum(out_[3, :] * out_[2, :]) / np.sum(out_[3, :])))


if __name__ == "__main__":
    main()