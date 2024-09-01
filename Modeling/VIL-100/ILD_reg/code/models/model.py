import cv2
import math
import copy

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from models.backbone import *
from libs.utils import *


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    pe = pe.view(1, d_model * 2, height, width)
    return pe


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, height, width):
        mask = torch.ones((1, height, width), dtype=torch.float32)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


class Deformable_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Deformable_Conv2d, self).__init__()
        self.deform_conv2d = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, offset, mask=None):
        x = self.deform_conv2d(x, offset, mask)
        return x


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


class Pixel_Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Pixel_Decoder, self).__init__()

        self.sf = cfg.scale_factor['img']
        self.seg_sf = cfg.scale_factor['seg']

        self.upsample_sf = [1, 2, 4]

        self.c_dims = cfg.c_dims
        self.num_stages = len(self.sf)

        in_channels = [128, 256, 512]

        self.feat_squeeze = nn.ModuleList([nn.Sequential(
            Conv_Norm_Act(in_channels[i], self.c_dims, kernel_size=3, padding=1, conv_type='2d'))
            for i in range(self.num_stages)])

        self.feat_combine = nn.Sequential(
            Conv_Norm_Act(self.c_dims * len(in_channels), self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            nn.Conv2d(self.c_dims, self.c_dims, 1)
        )

    def forward(self, features):
        for i in range(self.num_stages - 1, -1, -1):
            x = self.feat_squeeze[i](features[i])
            if self.upsample_sf[i] != 1:
                x = nn.functional.interpolate(x, scale_factor=self.upsample_sf[i], mode='bilinear', align_corners=False)
            features[i] = x

        x_concat = torch.cat(features, dim=1)
        x_combined = self.feat_combine(x_concat)
        out = nn.functional.interpolate(x_combined, scale_factor=2, mode='bilinear', align_corners=False)

        return out

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']

        self.c_dims = cfg.c_dims

        self.feat_embedding = nn.Sequential(
            Conv_Norm_Act(1, self.c_dims, kernel_size=3, padding=1, dilation=1, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=1, dilation=1, conv_type='2d'),
        )

        self.regressor = nn.Sequential(
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
        )

        kernel_size = 5
        self.offset_regression = nn.Sequential(
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            nn.Conv2d(self.c_dims, 2 * kernel_size * kernel_size, 1)
        )

        self.mask_regression = nn.Sequential(
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            nn.Conv2d(self.c_dims, kernel_size * kernel_size, 1)
        )

        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_dims, out_channels=self.cfg.top_m,
                                               kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        pos_embedding = PositionEmbeddingSine(num_pos_feats=self.c_dims // 2, normalize=True)
        self.pos_embeds = pos_embedding(self.cfg.height // self.seg_sf[0], self.cfg.width // self.seg_sf[0]).cuda()

    def forward_for_regression(self, input_tensor):
        feat_c = self.feat_embedding(input_tensor)
        feat_c = feat_c + self.pos_embeds

        offset = self.offset_regression(feat_c)
        mask = self.mask_regression(feat_c)

        x = self.regressor(feat_c)
        self.coeff_map = self.deform_conv2d(x, offset, mask)

        return {'coeff_map': self.coeff_map}