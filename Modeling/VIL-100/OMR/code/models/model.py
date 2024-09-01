import cv2
import math
import copy

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable

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

class Deformable_Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(Deformable_Conv2d, self).__init__()
        self.deform_conv2d = torchvision.ops.DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x, offset, mask=None):
        x = self.deform_conv2d(x, offset, mask)
        return x

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next, [f, i, g, o]

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state, return_gates=False):
        """

        Parameters
        ----------
        input_tensor: todo
            4-D Tensor either of shape (t, c, h, w)

        Returns
        -------
        last_state_list, layer_output
        """

        cur_layer_input = input_tensor

        all_gates = []
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            res = self.cell_list[layer_idx](cur_layer_input, cur_state=[h, c])
            if return_gates:
                h, c, gates = res
                all_gates.append(gates)
            else:
                h, c = res
            hidden_state[layer_idx] = h, c
            cur_layer_input = h

        if return_gates:
            return h, hidden_state, all_gates
        return h, hidden_state

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        # cfg
        self.cfg = cfg
        self.U = load_pickle(f'{self.cfg.dir["pre2"]}/U')[:, :self.cfg.top_m]

        self.sf = self.cfg.scale_factor['img']
        self.seg_sf = self.cfg.scale_factor['seg']

        self.c_dims = cfg.c_dims
        self.c_dims2 = cfg.c_dims2

        #####################################################################
        self.hidden_size = 64
        self.kernel_size = 3
        padding = 0 if self.kernel_size == 1 else 1

        self.dropout = 0.0
        self.skip_mode = 'concat'

        self.conv_lstm = ConvLSTM(input_dim=self.c_dims,
                                  hidden_dim=[self.c_dims, self.c_dims, self.c_dims],
                                  kernel_size=(3, 3),
                                  num_layers=3,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=True)

        self.lane_feat_embedding = nn.Sequential(
            Conv_Norm_Act(1, self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            # nn.Conv2d(self.c_dims, self.c_dims, kernel_size=1),
        )

        self.obj_feat_embedding = nn.Sequential(
            Conv_Norm_Act(1, self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            # nn.Conv2d(self.c_dims, self.c_dims, kernel_size=1),
        )

        self.combined_feat_embedding = nn.Sequential(
            Conv_Norm_Act(self.c_dims * 4, self.c_dims * 2, kernel_size=3, padding=1, conv_type='2d'),
            Conv_Norm_Act(self.c_dims * 2, self.c_dims, kernel_size=3, padding=1, conv_type='2d'),
            # nn.Conv2d(self.c_dims * 2, self.c_dims, kernel_size=1),
        )

        # classifier & regressor
        self.classifier = nn.Sequential(
            Conv_Norm_Act(self.c_dims, self.c_dims, kernel_size=1, conv_type='2d'),
            nn.Conv2d(self.c_dims, 2, kernel_size=1),
        )

        self.feat_embedding = nn.Sequential(
            Conv_Norm_Act(1, self.c_dims2, kernel_size=3, padding=1, dilation=1, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=1, dilation=1, conv_type='2d'),
        )

        self.regressor = nn.Sequential(
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
        )

        kernel_size = 5
        self.offset_regression = nn.Sequential(
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            nn.Conv2d(self.c_dims2, 2 * kernel_size * kernel_size, 1)
        )

        self.mask_regression = nn.Sequential(
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            Conv_Norm_Act(self.c_dims2, self.c_dims2, kernel_size=3, padding=2, dilation=2, conv_type='2d'),
            nn.Conv2d(self.c_dims2, kernel_size * kernel_size, 1)
        )

        self.deform_conv2d = Deformable_Conv2d(in_channels=self.c_dims2, out_channels=self.cfg.top_m,
                                               kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        pos_embedding = PositionEmbeddingSine(num_pos_feats=self.c_dims2 // 2, normalize=True)
        self.pos_embeds = pos_embedding(self.cfg.height // self.seg_sf[0], self.cfg.width // self.seg_sf[0]).cuda()

    def forward_for_ConvLSTM(self, input_tensor, hidden_state, return_gates=False):
        res = self.conv_lstm(input_tensor, hidden_state, return_gates)
        return res

    def forward_for_data_translation(self, is_training=True):
        out = dict()

        if self.prev_frame_num == 0:
            B, _, H, W = self.memory['img_feat']['t-0'].shape
            self.hidden_state = self.conv_lstm._init_hidden(batch_size=B, image_size=[H, W])
        else:
            img_feat = self.memory['img_feat'][f"t-{0}"]
            prev_img_feat = self.memory['img_feat'][f"t-{1}"]
            obj_mask = (self.memory['obj_mask'][f"t-{0}"] > 0.3).type(torch.float)
            lane_mask = self.memory['lane_mask'][f"t-{1}"]
            # lane_pos_map = self.memory['lane_pos_map'][f"t-{1}"]
            # lane_map = torch.cat((lane_mask, lane_pos_map), dim=1)

            lane_feat = self.lane_feat_embedding(lane_mask)
            obj_feat = self.obj_feat_embedding(obj_mask)

            feat_combined = self.combined_feat_embedding(torch.cat((img_feat, prev_img_feat, lane_feat, obj_feat), dim=1))

            # hidden_state_old = self.hidden_state[2]
            res = self.forward_for_ConvLSTM(feat_combined, self.hidden_state, return_gates=True)
            h_last, hidden_state, gates = res

            self.curr_img_feat_lstm = img_feat + h_last

            self.hidden_state = hidden_state

            return {'lstm': out,
                    'curr_img_feat': img_feat,
                    # 'prev_img_feat': prev_img_feat,
                    # 'convlstm_old_feat': hidden_state_old,
                    # 'convlstm_new_feat': hidden_state[2],
                    # 'feat_combined': feat_combined,
                    'curr_img_feat_lstm': self.curr_img_feat_lstm,
                    # 'gates': gates[-1]
                    }

        return {'lstm': out}

    def forward_for_classification(self, input_tensor):
        self.prob_map_logit = self.classifier(input_tensor)
        self.prob_map = F.softmax(self.prob_map_logit, dim=1)[:, 1:2]

        return {'prob_map_logit': self.prob_map_logit,
                'prob_map': self.prob_map}

    def forward_for_regression(self, input_tensor):
        feat_c = self.feat_embedding(input_tensor)
        feat_c = feat_c + self.pos_embeds

        offset = self.offset_regression(feat_c)
        mask = self.mask_regression(feat_c)

        x = self.regressor(feat_c)
        self.coeff_map = self.deform_conv2d(x, offset, mask)

        return {'coeff_map': self.coeff_map}
