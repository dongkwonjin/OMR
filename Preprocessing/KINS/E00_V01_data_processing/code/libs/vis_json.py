import os
import numpy as np
# import cvbase as cvb
import json
import cv2
from pycocotools.coco import COCO
import pycocotools.mask as mask_utils
import pdb
import copy


def polys_to_mask(polygons, height, width):
    # rles = mask_utils.frPyObjects(polygons, height, width)
    # rle = mask_utils.merge(rles)
    # mask = mask_utils.decode(rle)

    mask2 = np.zeros((height, width), dtype=np.uint8)
    mask2 = cv2.fillPoly(mask2, [np.int32(np.array(polygons[0]).reshape(-1, 2))], 1)
    return mask2


def vis_mask(img, a_m, i_m):
    cv2.namedWindow("a_i_m")
    a_i = copy.deepcopy(img)
    i_i = copy.deepcopy(img)

    a_m = a_m.astype(np.uint8) * 255
    i_m = i_m.astype(np.uint8) * 255
    a_m = np.stack((a_m, a_m, a_m), axis=2)
    i_m = np.stack((i_m, i_m, i_m), axis=2)

    a_m_w = cv2.addWeighted(a_i, 0.3, a_m, 0.7, 0)
    i_m_w = cv2.addWeighted(i_i, 0.3, i_m, 0.7, 0)
    a_i_m = np.concatenate((a_m_w, i_m_w), axis=0)

    cv2.imshow("a_i_m", a_i_m)
    cv2.waitKey(0)


def make_json_dict(imgs, anns):
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict


is_train = True

if is_train:
    base_img_path = "/media/dkjin/hdd1/Dataset/KINS/training/image_2/"
    base_ann_path = "/media/dkjin/hdd1/Dataset/KINS/update_train_2020.json"
else:
    base_img_path = "/Users/qilu/Documents/dataset/KITTI/Magicdata_7518_20180905/Magicdata_7518_20180905/img"
    base_ann_path = "/Users/qilu/Documents/update_test_20200216.json"

# anns = cvb.load(base_ann_path)
with open(base_ann_path) as f:
    anns = json.load(f)
imgs_info = anns['images']
anns_info = anns["annotations"]

imgs_dict, anns_dict = make_json_dict(imgs_info, anns_info)

for img_id in anns_dict.keys():
    img_name = imgs_dict[img_id]

    img_path = os.path.join(base_img_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    height, width, _ = img.shape
    anns = anns_dict[img_id]

    for ann in anns:
        a_mask = polys_to_mask(ann["a_segm"], height, width)
        i_mask = polys_to_mask(ann["i_segm"], height, width)

        vis_mask(img, a_mask, i_mask)