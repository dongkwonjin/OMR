import numpy as np
from libs.utils import *

class Video_Memory(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def forward_for_dict_initialization(self):
        self.keylist = ['img_feat', 'lane_mask', 'lane_pos_map', 'obj_mask']
        self.data = dict()
        for key in self.keylist:
            self.data[key] = dict()
        self.memory_t = 0

    def forward_for_dict_memorization(self):
        for i in range(self.memory_t - 1, -1, -1):
            for key in self.keylist:
                self.data[key][f't-{i+1}'] = self.data[key][f't-{i}']

        for key in self.keylist:
            self.data[key].pop('t-0')
        if self.memory_t >= self.cfg.num_t:
            self.memory_t -= 1

    def forward_for_dict_preparation(self):
        if self.prev_frame_num == 0:
            self.forward_for_dict_initialization()
        else:
            self.forward_for_dict_memorization()
        self.forward_for_dict_initialization_per_frame()

    def forward_for_dict_initialization_per_frame(self):
        for key in self.keylist:
            self.data[key][self.key_t] = dict()

    def forward_for_dict_update_per_frame(self, model, t_idx=None, mode=None, is_training=False):
        if mode == 'intra' and t_idx is not None and is_training == True:
            self.data['img_feat'][self.key_t] = model.img_feat[:, t_idx]
            self.data['obj_mask'][self.key_t] = model.obj_mask[:, t_idx]
            self.memory_t += 1

        elif mode == 'intra' and t_idx is not None and is_training == False:
            self.data['img_feat'][self.key_t] = model.img_feat[t_idx:t_idx + 1]
            self.data['obj_mask'][self.key_t] = model.obj_mask[t_idx:t_idx + 1]
            self.memory_t += 1

        elif mode == 'update' and t_idx is None:
            self.data['img_feat'][self.key_t] = model.curr_img_feat_lstm.detach()

    def forward_for_dict_transfer(self, model):
        model.memory = dict()
        for key in self.keylist:
            model.memory[key] = self.data[key]
        model.prev_frame_num = self.prev_frame_num
        return model

