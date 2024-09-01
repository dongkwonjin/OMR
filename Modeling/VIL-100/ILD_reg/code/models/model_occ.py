import cv2
import math
import copy

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from libs.utils import *

from transformers import SegformerForSemanticSegmentation

class Conv_Norm_Act(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False, norm='bn', act='relu',
                 conv_type='1d', conv_init='normal', norm_init=1.0):
        super(Conv_Norm_Act, self).__init__()
        if conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        if norm is not None:
            if conv_type == '1d':
                self.norm = nn.BatchNorm1d(out_channels)
            else:
                self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
        if act is not None:
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        return x

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']

        self.c_dims = cfg.c_dims2

        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").cuda()
        self.categ_idx = [11, 12, 13, 14, 15, 16, 17, 18]

    def forward_for_occlusion_detection(self, img):
        outputs = self.model(img)
        logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        prob_map = F.softmax(logits, dim=1)
        self.obj_mask = torch.sum(prob_map[:, self.categ_idx], dim=1, keepdim=True)
        # self.obj_mask = F.interpolate(self.obj_mask, scale_factor=1 / 2, mode='bilinear', align_corners=False)

        return {'obj_mask': self.obj_mask}
