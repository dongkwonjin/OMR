import os
import cv2
import torch
import math

import numpy as np
import torch.nn.functional as F

from libs.utils import *

class Query_Selector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def run_for_nms_interval(self):
        out = list()
        B, _, L = self.data.shape
        cls_row = self.data[:, 0]
        visit_all = torch.ones(B, L).type(torch.float).cuda()
        batch_idx = torch.arange(B).cuda()
        section = torch.linspace(0, L, self.cfg.num_query_selected + 1).type(torch.int)

        kernel_size = 3
        padding = (kernel_size - 1) // 2

        for i in range(self.cfg.num_query_selected):
            # selection
            st, ed = section[i], section[i + 1]
            idx_max = torch.argmax((cls_row * visit_all)[:, st:ed], dim=1) + st
            out.append(idx_max.view(-1, 1))

            r_mask = torch.zeros(B, L).type(torch.float).cuda()  # removal mask
            r_mask[batch_idx, idx_max] = 1
            r_mask[batch_idx, st:ed] = 1
            # r_mask = F.max_pool1d(r_mask.view(B, 1, L), kernel_size, stride=1, padding=padding)[:, 0]

            # removal
            visit_all = (1 - r_mask) * visit_all

        out = torch.cat(out, dim=1)
        return {'selected_query_idx': out}

    def run_for_nms(self):
        out = list()
        B, _, L = self.data.shape
        cls_row = self.data[:, 0]
        visit_all = torch.ones(B, L).type(torch.float).cuda()
        batch_idx = torch.arange(B).cuda()

        kernel_size = 5
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

        out = torch.cat(out, dim=1)
        out = torch.sort(out, dim=1)[0]
        return {'selected_query_idx': out}

    def run(self, data):
        # results
        out = dict()

        self.data = data
        out.update(self.run_for_nms())

        return out
