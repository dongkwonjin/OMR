import os
import cv2
import torch
import math

import numpy as np

from libs.utils import *

class Post_Processing(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def draw_polyline_cv(self, data, color=(255, 0, 0), s=1):
        out = np.ascontiguousarray(np.zeros(self.prob_map.shape, dtype=np.uint8))
        pts = np.int32(data).reshape((-1, 1, 2))
        out = cv2.polylines(out, [pts], False, color, s)
        return out

    def measure_confidence_score(self, prob_map, lane_mask):
        lane_mask[:self.height_idx[0]] = 0
        score = np.sum(lane_mask * prob_map) / np.sum(lane_mask)
        return score

    def set_vertical_range(self):
        self.idxlist = to_np(torch.sum((self.prob_map > self.cfg.height_thresd), dim=1)).nonzero()[0]
        if len(self.idxlist) > 0:
            idx_ed = self.idxlist[0] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            idx_st = self.idxlist[-1] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            lane_idx_ed = np.argmin(np.abs(self.cfg.py_coord - idx_ed))
            lane_idx_st = np.argmin(np.abs(self.cfg.py_coord - idx_st)) + 1
            self.height_idx = [self.idxlist[0], self.idxlist[-1]]
            return {'height_idx': [lane_idx_ed, lane_idx_st],
                    'seg_height_idx': self.height_idx}
        else:
            self.height_idx = [0]
            return {'height_idx': [0],
                    'seg_height_idx': self.height_idx}

    def coeff_to_x_coord_conversion(self, coeff):
        x_coords = list()

        if len(coeff) != 0:
            coeff = torch.cat(coeff, dim=1)
        if len(coeff) != 0:
            x_coords = torch.matmul(self.U, coeff)
            x_coords = x_coords * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
            x_coords = x_coords.permute(1, 0)

        self.x_coords = x_coords
        return {'x_coords': x_coords}

    def nms(self):
        out = dict()
        out['idx'] = []
        out['coeff'] = []

        prob_map = to_np(self.prob_map)
        coeff_map = self.coeff_map.clone()
        h, w = prob_map.shape

        for i in range(self.cfg.max_lane_num * 2):
            idx_max = np.argmax(prob_map)
            if len(out['idx']) >= self.cfg.max_lane_num:
                break
            if prob_map[idx_max // w, idx_max % w] > self.cfg.nms_thresd:
                coeff = coeff_map[:, idx_max // w, idx_max % w]
                # removal
                x_coords = torch.matmul(self.U, coeff) * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2
                x_coords = x_coords / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1)
                y_coords = to_tensor(self.cfg.py_coord).view(1, len(x_coords), 1) / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1)
                x_coords = x_coords.view(1, len(x_coords), 1)
                lane_coords = torch.cat((x_coords, y_coords), dim=2)
                lane_mask = self.draw_polyline_cv(to_np(lane_coords), color=(1, 1, 1), s=self.cfg.removal['lane_width'])
                lane_mask2 = self.draw_polyline_cv(to_np(lane_coords), color=(1, 1, 1), s=1)
                score = self.measure_confidence_score(prob_map, lane_mask2)
                prob_map[idx_max // w, idx_max % w] = 0
                if score >= 0.3:
                    out['idx'].append(int(idx_max))
                    out['coeff'].append(coeff.view(-1, 1))
                    prob_map = prob_map * (1 - lane_mask)
            else:
                break
        return out

    def draw_lane_mask(self, pts, color=(1, 1, 1), s=1):
        out = np.ascontiguousarray(self.lane_mask)
        out = cv2.polylines(out, np.int32(pts), False, color, s)
        return out

    def run_for_lane_mask_generation(self, data, prev_frame_num=None, is_training=False):
        out = list()
        out2 = list()
        for i in range(len(data)):
            N = len(data[i]['x_coords'])
            H, W = self.prob_map.shape
            self.lane_mask = np.zeros(self.prob_map.shape, dtype=np.uint8)
            self.lane_pos_map = np.zeros((H, W, self.cfg.top_m), dtype=np.float32)
            if N > 0:
                coeff = to_np(torch.cat(data[i]['coeff'], dim=1)[None])
                y_coords = to_tensor(self.cfg.py_coord).view(1, -1, 1)
                for j in range(len(data[i]['x_coords'])):
                    x_coords = data[i]['x_coords'][j:j + 1, :, None]
                    xy_coords = to_np(torch.cat((x_coords, y_coords), dim=2))

                    xy_coords[:, :, 0] = xy_coords[:, :, 0] / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1)
                    xy_coords[:, :, 1] = xy_coords[:, :, 1] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1)
                    self.draw_lane_mask(xy_coords, color=j + 1)
                    self.lane_pos_map[self.lane_mask == (j + 1), :] = coeff[:, :, j]  # coeff

                idxlist = data[i]['seg_height_idx']
                self.lane_mask = np.uint8(self.lane_mask != 0)
                self.lane_mask[:idxlist[0], :] = 0
                self.lane_pos_map[:idxlist[0], :] = 0

            # if is_training == True:
            #     if prev_frame_num == 0:
            #         self.lane_mask = to_np(self.seg_label[i] != 0)

            out.append(to_tensor(self.lane_mask).type(torch.float32)[None, None, :, :])
            out2.append(to_tensor(self.lane_pos_map).type(torch.float32)[None, :, :].permute(0, 3, 1, 2))
        lane_mask = torch.cat(out, dim=0)
        lane_pos_map = torch.cat(out2, dim=0)
        return {'lane_mask': lane_mask,
                'lane_pos_map': lane_pos_map}

    def run_for_nms(self, data, is_training=False):
        batch_out = list()
        b = len(data['prob_map'])

        for i in range(b):
            out = dict()
            self.prob_map = data['prob_map'][i, 0]
            self.coeff_map = data['coeff_map'][i]
            out.update(self.set_vertical_range())
            out.update(self.nms())
            out.update(self.coeff_to_x_coord_conversion(out['coeff']))
            batch_out.append(out)

        return batch_out
