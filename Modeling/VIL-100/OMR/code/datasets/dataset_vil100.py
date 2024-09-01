import cv2
import json
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from PIL import Image

from datasets.transforms import *
from libs.utils import *

class Dataset_Train(Dataset):
    def __init__(self, cfg, update=None):
        self.cfg = cfg
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_train"]}/datalist_{8}')
        self.datalist = list(self.datalist_video)

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.kins_data = load_pickle(f'{self.cfg.dir["kins_anno"]}/pickle/datalist')
        self.datalist_kins = list(self.kins_data)



    def generate_datalist(self):
        clip_length = self.cfg.clip_length
        random.shuffle(self.datalist)
        self.datalist_updated = list()
        self.flip_list = list()
        self.clip_idxlist = list()
        self.shift_list = list()
        self.kins_list = dict()
        self.kins_list['img_idx'] = list()
        self.kins_list['anno_idxlist'] = list()
        self.kins_list['ds'] = list()
        self.kins_list['dx'] = list()
        self.kins_list['dy'] = list()
        self.kins_list['ts'] = list()
        self.kins_list['tx'] = list()
        self.kins_list['ty'] = list()
        clip_idxlist = list(np.arange(0, clip_length))
        for i in range(len(self.datalist)):
            name = self.datalist[i]
            clip_list = sorted(random.sample(self.datalist_video[name]['past'], clip_length - 1)) + self.datalist_video[name]['current']
            self.datalist_updated += clip_list
            self.flip_list += [random.randint(0, 1)] * clip_length
            self.clip_idxlist += clip_idxlist
            self.shift_list += np.arange(0, len(clip_list)).tolist()

            img_idx = self.datalist_kins[random.randint(0, len(self.datalist_kins) - 1)]
            num_obj = np.minimum(random.randint(1, 3), len(self.kins_data[img_idx]['annotations']))
            anno_idxlist = random.sample(range(0, len(self.kins_data[img_idx]['annotations'])), num_obj)
            self.kins_list['img_idx'] += [img_idx] * clip_length
            self.kins_list['anno_idxlist'] += [anno_idxlist] * clip_length

            self.kins_list['ds'] += [random.uniform(-0.01, 0.01)] * clip_length
            self.kins_list['dx'] += [random.uniform(-30, 30)] * clip_length
            self.kins_list['dy'] += [random.uniform(-30,  30)] * clip_length
            self.kins_list['ts'] += [random.uniform(0, 0.5)] * clip_length
            self.kins_list['tx'] += [random.uniform(-50, 50)] * clip_length
            self.kins_list['ty'] += [random.uniform(-50, 50)] * clip_length

        idx_end = (len(self.datalist) // clip_length) * clip_length
        self.datalist_updated = self.datalist_updated[:idx_end]
        self.flip_list = self.flip_list[:idx_end]
        self.clip_idxlist = self.clip_idxlist[:idx_end]
        self.shift_list = self.shift_list[:idx_end]
        self.kins_list['img_idx'] = self.kins_list['img_idx'][:idx_end]
        self.kins_list['anno_idxlist'] = self.kins_list['anno_idxlist'][:idx_end]
        self.kins_list['ds'] = self.kins_list['ds'][:idx_end]
        self.kins_list['dx'] = self.kins_list['dx'][:idx_end]
        self.kins_list['dy'] = self.kins_list['dy'][:idx_end]
        self.kins_list['ts'] = self.kins_list['ts'][:idx_end]
        self.kins_list['tx'] = self.kins_list['tx'][:idx_end]
        self.kins_list['ty'] = self.kins_list['ty'][:idx_end]

    def cropping(self, img, lanes):
        if img is not None:
            img = img[self.cfg.crop_size:]
            if self.flip == 1:
                img = cv2.flip(img, 1)

        for i in range(len(lanes)):
            if len(lanes[i]) == 0:
                continue
            lanes[i][:, 1] -= self.cfg.crop_size
            if self.flip == 1:
                lanes[i][:, 0] = (self.cfg.width - 1) - lanes[i][:, 0]

        return img, lanes

    def get_data_org(self, file_name):
        img = Image.open(f'{self.cfg.dir["dataset"]}/JPEGImages/{file_name}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir["pre0_train"]}/{file_name}')
        return img, anno


    def get_overlap_data(self, lane_pts, idx):
        self.overlap_mask = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)[:, :, None]
        self.overlap_img = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        img_idx = self.kins_list['img_idx'][idx]
        anno_idxlist = self.kins_list['anno_idxlist'][idx]

        ds = self.kins_list['ds'][idx]
        dx = self.kins_list['dx'][idx]
        dy = self.kins_list['dy'][idx]
        ts = self.kins_list['ts'][idx]
        tx = self.kins_list['tx'][idx]
        ty = self.kins_list['ty'][idx]

        if len(lane_pts) != 0:
            # lane_idx = random.randint(0, len(lane_pts) - 1)
            lane_cx, lane_cy = np.mean(np.concatenate(lane_pts, axis=0), axis=0)

        A = np.identity(2)
        b = np.array([[dx], [dy]])
        M = np.concatenate((A, b), axis=1)

        img_name = self.kins_data[img_idx]['file_name']
        img = Image.open(f'{self.cfg.dir["kins_img"]}/{img_name}').convert('RGB')

        for i in range(len(anno_idxlist)):
            anno_idx = anno_idxlist[i]
            if self.kins_data[img_idx]['annotations'][anno_idx]['category_id'] == 5:  # categ : tram
                continue
            anno_seg = self.kins_data[img_idx]['annotations'][anno_idx]['a_segm'][0]
            anno_seg = [np.float32(anno_seg).reshape(-1, 2)]

            overlap_img, anno_new = self.transform.process_for_syn_generation(img, anno_seg)
            overlap_mask = np.zeros((self.cfg.height, self.cfg.width), dtype=np.uint8)
            overlap_mask = cv2.fillPoly(overlap_mask, [np.int32(anno_new)], 1)

            H, W, _ = overlap_img.shape

            obj_cx, obj_cy = np.mean(anno_new[0], axis=0)
            if len(lane_pts) == 0:
                lane_cx, lane_cy, obj_cx, obj_cy = 0, 0, 0, 0

            M[0, 0] += (ts + ds * self.clip_idx)
            M[1, 1] += (ts + ds * self.clip_idx)
            M[0, 2] += ((lane_cx - obj_cx) + tx + dx * self.clip_idx)
            M[1, 2] += ((lane_cy - obj_cy) + ty + dy * self.clip_idx)
            overlap_img = cv2.warpAffine(overlap_img, M, (W, H))
            overlap_mask = cv2.warpAffine(overlap_mask, M, (W, H))

            self.overlap_img[overlap_mask == 1] = overlap_img[overlap_mask == 1]
            self.overlap_mask += overlap_mask[:, :, None]

        self.overlap_mask = np.uint8(self.overlap_mask != 0)
    def get_data_aug(self, img, anno, idx):
        img_new, anno_new = self.transform.process(img, anno['lane'])
        img_new, anno_new = self.cropping(img_new, anno_new)

        if self.clip_idx != 0:
            self.get_overlap_data(anno_new, idx)
            img_new = img_new * (1 - self.overlap_mask) + self.overlap_img * self.overlap_mask

        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size

        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lanes': anno_new,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_data_preprocessed(self, data):
        # initialize error case
        self.transform.init_error_case()
        self.transform.img_name = data['img_name']

        # preprocessing
        out = dict()
        out.update(self.transform.get_lane_components(data['lanes']))
        out.update(self.transform.approximate_lanes(out['extended_lanes']))
        out['is_error_case'] = self.transform.is_error_case['total']
        return out

    def get_downsampled_label_seg(self, lanes, idx, sf):
        for s in sf:
            lane_pts = np.copy(lanes)
            lane_pts[:, 0] = lanes[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
            lane_pts[:, 1] = lanes[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)
            self.label['seg_label'][s] = cv2.polylines(self.label['seg_label'][s], [np.int32(lane_pts)], False, idx + 1, self.cfg.lane_width['seg'])

    def get_downsampled_label_coeff(self, data, idx, sf):
        for s in sf:
            self.label['coeff_label'][s][self.label['seg_label'][s] == (idx + 1), :] = data

    def get_label_map(self, data):
        out = dict()

        self.label = dict()
        self.label['org_label'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32))
        self.label['seg_label'] = dict()
        self.label['coeff_label'] = dict()

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.ascontiguousarray(np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.uint8))
            self.label['coeff_label'][s] = np.zeros((self.cfg.height // s, self.cfg.width // s, self.cfg.top_m), dtype=np.float32)

        for i in range(len(data['lanes'])):
            lane_pts = data['lanes'][i]
            self.label['org_label'] = cv2.polylines(self.label['org_label'], [np.int32(lane_pts)], False, i + 1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)
            self.get_downsampled_label_seg(lane_pts, i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = cv2.dilate(self.label['seg_label'][s], kernel=(3, 3), iterations=1)

        for i in range(len(data['lanes'])):
            self.get_downsampled_label_coeff(data['c'][i], i, self.cfg.scale_factor['seg'])

        self.label['org_label'] = np.float32(self.label['org_label'] != 0)

        out.update(self.label)

        return out

    def get_position_for_query_selection(self):
        out = dict()
        idxlist = np.sum(self.label['seg_label'][self.cfg.scale_factor['seg'][0]], axis=1).nonzero()[0]
        if len(idxlist) == 0:
            out['pad'] = self.cfg.pad['ed'][:2]
            out['check'] = 0
        else:
            out['pad'] = [self.label['seg_label'][self.cfg.scale_factor['seg'][0]].shape[0] - int(np.median(idxlist)), self.cfg.pad["ed"][1]]
            out['check'] = 1

        return out

    def get_coeff_label(self, data):
        out = dict()

        coeff = data['c']
        num_lanes = len(coeff)
        data.pop('c')
        out['c'] = np.zeros((self.cfg.max_lane_num * 2, self.cfg.top_m), np.float32)
        if num_lanes > 0:
            out['c'][:num_lanes] = coeff
        out['num_lanes'] = num_lanes

        return out

    def get_height_label(self, data):
        out = dict()

        out['height'] = np.zeros((self.cfg.max_lane_num * 2, 1), np.float32)
        for i in range(len(data['lanes'])):
            out['height'][i] = np.min(data['lanes'][i][:, 1]) / (self.cfg.height - 1)

        return out


    def remove_dict_keys(self, data):
        keylist = ['lanes', 'extended_lanes', 'approx_lanes']
        for key in keylist:
            data.pop(key)
        return data

    def __getitem__(self, idx):
        out = dict()
        self.flip = self.flip_list[idx]
        self.clip_idx = self.clip_idxlist[idx]
        self.shift_idx = self.shift_list[idx]
        out['img_name'] = self.datalist_updated[idx]
        img, anno = self.get_data_org(out['img_name'])
        out.update(self.get_data_aug(img, anno, idx))
        out.update(self.get_data_preprocessed(out))
        out.update(self.get_label_map(out))
        out.update(self.get_coeff_label(out))
        out.update(self.get_height_label(out))
        out.update(self.get_position_for_query_selection())
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)

class Dataset_Test(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg

        self.datalist = load_pickle(f'{self.cfg.dir["pre0_test"]}/datalist')
        self.datalist_video = load_pickle(f'{self.cfg.dir["pre3_test"]}/datalist_{8}')

        if cfg.sampling == True:
            sampling = np.arange(0, len(self.datalist), cfg.sampling_step)
            self.datalist = np.array(self.datalist)[sampling].tolist()

        # image transform
        self.transform = Transforms(cfg)
        self.transform.settings()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def cropping(self, img, lanes):
        img = img[self.cfg.crop_size:]
        for i in range(len(lanes)):
            lanes[i][:, 1] -= self.cfg.crop_size
        return img, lanes

    def get_data_org(self, idx):
        img = Image.open(f'{self.cfg.dir["dataset"]}/JPEGImages/{self.datalist[idx]}.jpg').convert('RGB')
        anno = load_pickle(f'{self.cfg.dir[f"pre0_test"]}/{self.datalist[idx]}')
        return img, anno

    def get_data_aug(self, img, anno):
        img_new, anno_new = self.transform.process_for_test(img, anno['lane'])
        img_new, anno_new = self.cropping(img_new, anno_new)

        img_new = Image.fromarray(img_new)
        img_new = self.to_tensor(img_new)
        self.org_width, self.org_height = img.size

        return {'img': self.normalize(img_new),
                'img_rgb': img_new,
                'lanes': anno_new,
                'org_h': self.org_height, 'org_w': self.org_width}

    def get_downsampled_label_seg(self, lanes, idx, sf):
        for s in sf:
            lane_pts = np.copy(lanes)
            lane_pts[:, 0] = lanes[:, 0] / (self.cfg.width - 1) * (self.cfg.width // s - 1)
            lane_pts[:, 1] = lanes[:, 1] / (self.cfg.height - 1) * (self.cfg.height // s - 1)

            self.label['seg_label'][s] = cv2.polylines(self.label['seg_label'][s], [np.int32(lane_pts)], False, idx + 1, self.cfg.lane_width['seg'])

    def get_seg_label(self, data):
        out = dict()

        self.label = dict()
        self.label['org_label'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32))
        self.label['seg_label'] = dict()

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.ascontiguousarray(np.zeros((self.cfg.height // s, self.cfg.width // s), dtype=np.uint8))

        for i in range(len(data['lanes'])):
            lane_pts = data['lanes'][i]
            self.label['org_label'] = cv2.polylines(self.label['org_label'], [np.int32(lane_pts)], False, i + 1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)
            self.get_downsampled_label_seg(lane_pts, i, self.cfg.scale_factor['seg'])

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = cv2.dilate(self.label['seg_label'][s], kernel=(3, 3), iterations=1)

        for s in self.cfg.scale_factor['seg']:
            self.label['seg_label'][s] = np.int64(self.label['seg_label'][s] != 0)
        self.label['org_label'] = np.float32(self.label['org_label'] != 0)

        out.update(self.label)

        return out

    def get_seg_label_future(self, img, idx):
        out = dict()
        anno = dict()
        if len(self.datalist_video[self.datalist[idx]]['future']) != 0:
            img_name = self.datalist_video[self.datalist[idx]]['future'][0]
            anno = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/label/validation/{img_name}')
            _, anno = self.cropping(None, anno)
            anno = self.transform.process_for_test_anno_only(img, anno['lanes'])

        self.label = dict()
        self.label['org_label_future'] = np.ascontiguousarray(np.zeros((self.cfg.height, self.cfg.width), dtype=np.float32))

        for i in range(len(anno)):
            lane_pts = anno[i]
            self.label['org_label_future'] = cv2.polylines(self.label['org_label_future'], [np.int32(lane_pts)], False, i + 1, self.cfg.lane_width['org'], lineType=cv2.LINE_AA)

        self.label['org_label_future'] = np.float32(self.label['org_label_future'] != 0)
        out.update(self.label)

        return out

    def remove_dict_keys(self, data):
        data.pop('lanes')
        return data

    def __getitem__(self, idx):
        out = dict()
        out['img_name'] = self.datalist[idx]
        out['prev_num'] = len(self.datalist_video[self.datalist[idx]]['past'])
        img, anno = self.get_data_org(idx)
        out.update(self.get_data_aug(img, anno))
        out.update(self.get_seg_label(out))
        # out.update(self.get_seg_label_future(img, idx))
        out = self.remove_dict_keys(out)

        return out

    def __len__(self):
        return len(self.datalist)