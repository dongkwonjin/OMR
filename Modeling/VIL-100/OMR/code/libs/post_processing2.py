import os
import cv2
import torch
import math

import numpy as np
import torch.nn.functional as F

from libs.utils import *

class Post_Processing(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]
        self.mask = np.ones((self.cfg.height // self.cfg.scale_factor['seg'][0], self.cfg.width // self.cfg.scale_factor['seg'][0]), dtype=np.int32)

    def draw_polyline_cv(self, pts, color=(1, 1, 1), s=1):
        out = np.ascontiguousarray(self.lane_mask)
        out = cv2.polylines(out, np.int32(pts), False, color, s)
        return out

    def run_for_coeff_to_x_coord_conversion(self):
        x_coords = list()
        if len(self.query_coeff) != 0:
            x_coords = torch.matmul(self.query_coeff, self.U.permute(1, 0))
            x_coords = x_coords * ((self.cfg.width - 1) / 2) + (self.cfg.width - 1) / 2

        return {'x_coords': x_coords}

    def run_for_height_determination(self):
        self.idxlist = to_np(torch.sum((self.prob_map[0] > self.cfg.height_thresd), dim=1) > 2).nonzero()[0]
        if len(self.idxlist) > 0:
            idx_ed = self.idxlist[0] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            idx_st = self.idxlist[-1] / (self.cfg.height // self.sf[0] - 1) * (self.cfg.height - 1)
            lane_idx_ed = np.argmin(np.abs(self.cfg.py_coord - idx_ed))
            lane_idx_st = np.argmin(np.abs(self.cfg.py_coord - idx_st)) + 1
            self.height_idx = [self.idxlist[0], self.idxlist[-1]]
            return {'height_idx': [lane_idx_ed, lane_idx_st],
                    'seg_height_idx': self.height_idx}
        else:
            self.height_idx = [0, -1]
            return {'height_idx': [0, -1],
                    'seg_height_idx': self.height_idx}

    def run_for_lane_mask_generation(self, data, prev_frame_num=None, is_training=False):
        out = list()
        out2 = list()
        for i in range(len(data)):
            N = len(data[i]['x_coords'])
            H, W = self.prob_map[0].shape
            self.lane_mask = np.zeros(self.prob_map[0].shape, dtype=np.uint8)
            self.lane_pos_map = np.zeros((H, W, self.cfg.top_m), dtype=np.float32)

            if N > 0:
                coeff = to_np(data[i]['coeff'])[:, None, :]
                y_coords = to_tensor(self.cfg.py_coord).view(1, -1, 1)
                for j in range(len(data[i]['x_coords'])):
                    x_coords = data[i]['x_coords'][j:j+1, :, None]
                    xy_coords = to_np(torch.cat((x_coords, y_coords), dim=2))

                    xy_coords[:, :, 0] = xy_coords[:, :, 0] / (self.cfg.width - 1) * (self.cfg.width // self.sf[0] - 1)
                    xy_coords[:, :, 1] = xy_coords[:, :, 1] / (self.cfg.height - 1) * (self.cfg.height // self.sf[0] - 1)
                    self.draw_polyline_cv(xy_coords, color=j + 1)
                self.lane_mask = cv2.dilate(self.lane_mask, kernel=(3, 3), iterations=1)
                for j in range(len(data[i]['x_coords'])):
                    self.lane_pos_map[self.lane_mask == (j + 1), :] = coeff[j]  # coeff

                idxlist = data[i]['seg_height_idx']
                self.lane_mask = np.uint8(self.lane_mask != 0)
                self.lane_mask[:idxlist[0], :] = 0
                self.lane_mask[idxlist[-1]:, :] = 0
                self.lane_pos_map[:idxlist[0], :] = 0
                self.lane_pos_map[idxlist[-1]:, :] = 0

            if is_training == True:
                if prev_frame_num == 0:
                    self.lane_mask = to_np(self.seg_label[i] != 0)
                    self.lane_pos_map = to_np(self.coeff_label[i])

            out.append(to_tensor(self.lane_mask).type(torch.float32)[None, None, :, :])
            out2.append(to_tensor(self.lane_pos_map).type(torch.float32)[None, :, :].permute(0, 3, 1, 2))
        lane_mask = torch.cat(out, dim=0)
        lane_pos_map = torch.cat(out2, dim=0)
        return {'lane_mask': lane_mask,
                'lane_pos_map': lane_pos_map}

    def set_vertical_range(self, prob_map):
        batch_pad = list()
        b, _, h, w = prob_map.shape
        h_map = np.sum((to_np(prob_map[:, 0]) > self.cfg.height_thresd), axis=2)
        for i in range(b):
            idxlist = h_map[i].nonzero()[0]
            if len(idxlist) == 0:
                pad = self.cfg.pad['ed'][:2]
            else:
                pad = [h - int(np.median(idxlist)), self.cfg.pad["ed"][1]]

            l = self.cfg.pad["ed"][2]
            b_left = self.mask[-pad[0] - l:-pad[0], pad[1]]
            b_bottom = self.mask[-pad[0], pad[1] + 1:-pad[1] - 1]
            b_right = self.mask[-pad[0] - l:-pad[0], -pad[1] - 1]
            if sum(b_left) + sum(b_bottom) + sum(b_right) != self.cfg.num_queries:
                pad = self.cfg.pad['ed'][:2]

            batch_pad.append(pad)
        batch_pad = np.array(batch_pad).transpose(1, 0)
        self.pad = batch_pad
        return {"boundary_pad": batch_pad}

    def search_region_selection(self, prob_map, coeff_map):
        x = torch.cat((prob_map.detach(), coeff_map.detach()), dim=1)

        pad = self.pad
        l = self.cfg.pad["ed"][2]
        b, _, _, _ = x.shape

        batch_row_data = list()
        for i in range(b):
            b_left = x[i:i + 1, :, -pad[0][i] - l:-pad[0][i], pad[1][i]]
            b_bottom = x[i:i + 1, :, -pad[0][i], pad[1][i]:-pad[1][i]]
            b_right = x[i:i + 1, :, -pad[0][i] - l:-pad[0][i], -pad[1][i] - 1]
            b_right = torch.flip(b_right, dims=[2])
            row_data = torch.cat((b_left, b_bottom, b_right), dim=2)
            batch_row_data.append(row_data)
        batch_row_data = torch.cat(batch_row_data, dim=0)

        return {'query_init': batch_row_data}

    def nms(self, data):
        out = list()
        B, _, L = data.shape
        cls_row = data[:, 0]
        visit_all = torch.ones(B, L).type(torch.float).cuda()
        batch_idx = torch.arange(B).cuda()

        kernel_size = 9
        padding = (kernel_size - 1) // 2
        for i in range(self.cfg.num_query_selected):
            # selection
            idx_max = torch.argmax(cls_row * visit_all, dim=1)
            out.append(idx_max.view(-1, 1))

            r_mask = torch.zeros(B, L).type(torch.float).cuda()  # removal mask
            r_mask[batch_idx, idx_max] = 1
            r_mask = F.max_pool1d(r_mask.view(B, 1, L), kernel_size, stride=1, padding=padding)[:, 0]

            # removal
            visit_all = (1 - r_mask) * visit_all

        selected_idx = torch.cat(out, dim=1)
        selected_idx = torch.sort(selected_idx, dim=1)[0]

        batch_idx = torch.arange(B).cuda()
        selected_lanes = data[batch_idx, :, selected_idx.permute(1, 0)].permute(1, 0, 2)

        return {'selected_lanes': selected_lanes}

    def run_for_nms(self, data):
        out = dict()
        prob_map = data['prob_map']
        coeff_map = data['coeff_map']
        out.update(self.set_vertical_range(prob_map))
        out.update(self.search_region_selection(prob_map, coeff_map))
        out.update(self.nms(out['query_init']))
        return out

    def run(self, data, is_training=False):
        batch_out = list()
        for i in range(len(data['prob_map'])):
            out = dict()
            self.prob_map = data['prob_map'][i]
            self.query_coeff = data['selected_lanes'][i][:, 1:1+self.cfg.top_m]
            self.query_prob = data['selected_lanes'][i][:, 0]

            # positive lanes
            idx_pos = torch.where(self.query_prob > self.cfg.prob_thresd)[0]
            # negative lanes
            # idx_neg = torch.where(self.query_prob < self.cfg.prob_thresd)[0]
            # idx_neg_selected = idx_neg[torch.randperm(len(idx_neg))][0:1]
            # selected
            # idx_guide = torch.cat((idx_pos, idx_neg_selected))
            idx_guide = idx_pos

            self.query_coeff = self.query_coeff[idx_guide]
            out['coeff'] = self.query_coeff
            out.update(self.run_for_height_determination())
            out.update(self.run_for_coeff_to_x_coord_conversion())
            batch_out.append(out)

        return batch_out