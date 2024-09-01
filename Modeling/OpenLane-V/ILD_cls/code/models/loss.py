import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.measure import regionprops

from libs.utils import *

class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll1 = nn.NLLLoss(reduce=True)
        self.nll2 = nn.NLLLoss(reduce=False)

    def forward(self, logits, labels, reduce=True):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score

        if reduce == True:
            loss = self.nll1(log_score, labels)
        else:
            loss = self.nll2(log_score, labels)
        return loss

class Loss_Function(nn.Module):
    def __init__(self, cfg):
        super(Loss_Function, self).__init__()
        self.cfg = cfg

        self.loss_mse = nn.MSELoss()
        self.loss_bce = nn.BCELoss()
        self.loss_nce = nn.CrossEntropyLoss()
        self.loss_score = nn.MSELoss()
        self.loss_focal = SoftmaxFocalLoss(gamma=2)

        self.sf = cfg.scale_factor['seg']
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

    def forward(self, out, gt):
        loss_dict = dict()

        seg_label = (gt['seg_label'][self.sf[0]] != 0).type(torch.long)
        loss_dict['seg'] = self.loss_focal(out['prob_map_logit'], seg_label) * 1e1
        # loss_dict['coeff_iou'] = self.compute_IoU_loss(out['coeff_map'].permute(0, 2, 3, 1), gt['coeff_label'][self.sf[0]], exclude_map=seg_label)

        occ_seg_label = (out['obj_mask'][:, 0] > 0.3).type(torch.long)
        loss_dict['occ'] = self.loss_focal(out['occ_prob_map_logit'], occ_seg_label)

        l_sum = torch.FloatTensor([0.0]).cuda()
        for key in list(loss_dict):
            l_sum += loss_dict[key]
        loss_dict['sum'] = l_sum

        return loss_dict

    def compute_IoU_loss(self, out, gt, exclude_map):
        e1 = 0.1
        out_x_coords = self.coeff_to_x_coord_conversion(out)
        gt_x_coords = self.coeff_to_x_coord_conversion(gt)

        d1 = torch.min(out_x_coords + e1, gt_x_coords + e1) - torch.max(out_x_coords - e1, gt_x_coords - e1)
        d2 = torch.max(out_x_coords + e1, gt_x_coords + e1) - torch.min(out_x_coords - e1, gt_x_coords - e1)

        d1 = torch.sum(d1, dim=3)
        d2 = torch.sum(d2, dim=3)

        iou = (d1 / (d2 + 1e-9))
        iou_loss = torch.mean((1 - iou)[exclude_map == 1])
        if torch.isnan(iou_loss):
            iou_loss = torch.FloatTensor([0.0]).cuda()

        return iou_loss

    def coeff_to_x_coord_conversion(self, coeff_map, mode=None):
        m = self.cfg.top_m
        b, h, w, _ = coeff_map.shape
        coeff_map = coeff_map.reshape(-1, m, 1)
        U = self.U.view(1, -1, m).expand(coeff_map.shape[0], -1, m)
        x_coord_map = torch.bmm(U, coeff_map)
        x_coord_map = x_coord_map.view(b, h, w, -1)
        return x_coord_map

    def compute_IoU_loss2(self, out, gt, exclude_map):
        e1 = 0.1
        out_x_coords = self.coeff_to_x_coord_conversion2(out)
        gt_x_coords = self.coeff_to_x_coord_conversion2(gt)

        d1 = torch.min(out_x_coords + e1, gt_x_coords + e1) - torch.max(out_x_coords - e1, gt_x_coords - e1)
        d2 = torch.max(out_x_coords + e1, gt_x_coords + e1) - torch.min(out_x_coords - e1, gt_x_coords - e1)

        d1 = torch.sum(d1, dim=2)
        d2 = torch.sum(d2, dim=2)

        iou = (d1 / (d2 + 1e-9))
        iou_loss = torch.mean((1 - iou)[exclude_map == 1])
        if torch.isnan(iou_loss):
            iou_loss = torch.FloatTensor([0.0]).cuda()

        return iou_loss

    def coeff_to_x_coord_conversion2(self, coeff_map):
        b, n, m = coeff_map.shape
        coeff_map = coeff_map.reshape(-1, m, 1)
        U = self.U.view(1, -1, m).expand(coeff_map.shape[0], -1, m)
        x_coord_map = torch.bmm(U, coeff_map)
        x_coord_map = x_coord_map.view(b, n, -1)
        return x_coord_map

    def get_row_label(self, gt_seg, gt_coeff):
        gt_row_seg = self.boundary_region_extraction(gt_seg)
        gt_row_coeff = self.boundary_region_extraction(gt_coeff)
        # gt_row_seg = self.get_centorids(to_np(gt_row_seg))
        # gt_row_coeff = gt_row_coeff * gt_row_seg
        gt_row_seg = (gt_row_seg != 0).type(torch.long)
        return gt_row_seg, gt_row_coeff

    def boundary_region_extraction(self, x):
        pad = self.pad
        l = self.cfg.pad["ed"][2]
        b, _, _, _ = x.shape
        batch_row_data = list()
        for i in range(b):
            b_left = x[i:i+1, :, -pad[0][i]-l:-pad[0][i], pad[1][i]]
            b_bottom = x[i:i+1, :, -pad[0][i], pad[1][i]:-pad[1][i]]
            b_right = x[i:i+1, :, -pad[0][i]-l:-pad[0][i], -pad[1][i] - 1]
            b_right = torch.flip(b_right, dims=[2])
            row_data = torch.cat((b_left, b_bottom, b_right), dim=2)

            batch_row_data.append(row_data)
        batch_row_data = torch.cat(batch_row_data, dim=0)
        return batch_row_data

    def get_centorids(self, data):
        b, _, n = data.shape
        out = np.zeros((b, 1, n), dtype=np.int64)
        for i in range(b):
            results = regionprops(data[i])
            for j in range(len(results)):
                centroid = results[j].centroid
                out[i, :, round(centroid[1])] = 1
        return to_tensor(out)