import os
import cv2
import torch
import math

import numpy as np
import torch.nn.functional as F

from libs.utils import *

class Evaluation_Flow(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.sf = cfg.scale_factor['seg']

    def measure_IoU(self, X1, X2):
        ep = 1e-7
        X = X1 + X2
        X_uni = torch.sum(X != 0, dim=(1, 2)).type(torch.float32)
        X_inter = torch.sum(X == 2, dim=(1, 2)).type(torch.float32)
        iou = X_inter / (X_uni + ep)
        return iou

    def run_for_fscore(self):
        if self.prev_frame_num < self.cfg.num_t - 1:
            return

        table = (self.warped_prev_map + 1) * (self.current_map + 2)
        self.results['tp'] += list(to_np(torch.sum(table == 6, dim=(1, 2))))
        self.results['tn'] += list(to_np(torch.sum(table == 2, dim=(1, 2))))
        self.results['fp'] += list(to_np(torch.sum(table == 4, dim=(1, 2))))
        self.results['fn'] += list(to_np(torch.sum(table == 3, dim=(1, 2))))

    def measure(self):
        ep = 1e-7
        results = load_pickle(f'{self.cfg.dir["out"]}/{self.mode}/pickle/eval_flow_results')

        tp = np.sum(np.float32(results['tp']))
        fp = np.sum(np.float32(results['fp']))
        fn = np.sum(np.float32(results['fn']))
        precision = tp / (tp + fp + ep)
        recall = tp / (tp + fn + ep)
        fscore = 2 * precision * recall / (precision + recall + ep)

        print(f'\nFLOW : precision {precision}, recall {recall}, fscore {fscore}\n')
        return {'flow_precision': precision, 'flow_recall': recall, 'flow_fscore': fscore}

    def init(self):
        self.results = dict()
        self.results['tp'] = list()
        self.results['tn'] = list()
        self.results['fp'] = list()
        self.results['fn'] = list()
        self.results['precision'] = 0
        self.results['recall'] = 0
        self.results['fscore'] = 0

    def update(self, batch, out, batch_idx=None, prev_frame_num=None, mode=None):
        self.mode = mode
        self.prev_frame_num = prev_frame_num
        if prev_frame_num < self.cfg.num_t - 1:
            self.prev_map = batch['seg_label'][self.sf[0]][batch_idx:batch_idx + 1].cuda()
        else:
            self.current_map = batch['seg_label'][self.sf[0]][batch_idx:batch_idx+1].cuda()
            _, h, w = self.prev_map.shape
            self.prev_map = self.prev_map.view(1, 1, h, w).type(torch.float32)
            grid = out['grid']
            self.warped_prev_map = F.grid_sample(self.prev_map, grid, mode='bilinear', padding_mode='zeros')[0]
            self.warped_prev_map = (self.warped_prev_map > 0.5).type(torch.float32)
            self.prev_map = self.current_map.clone()
