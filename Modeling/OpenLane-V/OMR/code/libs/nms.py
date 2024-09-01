import os
import cv2
import torch
import math

import numpy as np
import torch.nn.functional as F

from libs.utils import *

class Non_Maximum_Suppression(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

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
