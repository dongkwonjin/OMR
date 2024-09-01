import os
import torch

import numpy as np

class Config(object):
    def __init__(self):
        # --------basics-------- #
        self.setting_for_system()
        self.setting_for_path()
        self.setting_for_image_param()
        self.setting_for_dataloader()
        self.setting_for_visualization()
        self.setting_for_save()
        # --------preprocessing-------- #
        self.setting_for_preprocessing()
        # --------modeling-------- #
        self.setting_for_training()
        self.setting_for_postprocessing()
        self.setting_for_evaluation()
        self.setting_for_modeling()

    def setting_for_preprocessing(self):
        self.setting_for_lane_representation()
        self.setting_for_svd()
        # --------others-------- #

    def setting_for_system(self):
        self.gpu_id = "1"
        self.seed = 123

    def setting_for_path(self):
        self.pc = 'main'
        self.dir = dict()

        self.setting_for_dataset_path()  # dataset path

        self.dir['proj'] = os.path.dirname(os.getcwd()) + '/'
        # ------------------- need to modify ------------------- #
        self.dir['head_pre'] = '--preprocessed data path'
        # ------------------------------------------------------ #
        self.dir['pre2'] = f'{self.dir["head_pre"]}/E02_V01_SVD/output_training/pickle'
        self.dir['model1'] = f'{os.path.dirname(os.path.dirname(self.dir["proj"]))}/ILD_cls/output'

        self.dir['out'] = f'{os.getcwd().replace("code_", "output_")}'
        self.dir['weight'] = f'{self.dir["out"]}/train/weight'
        self.dir['pretrained_weight'] = f'{self.dir["model1"]}/train/weight'

    def setting_for_dataset_path(self):
        self.dataset_name = 'openlane'  # ['tusimple', 'vil100']
        self.datalist = 'training'  # ['train'] only

        # ------------------- need to modify ------------------- #
        self.dir['dataset'] = '--dataset path'
        # ------------------------------------------------------ #

    def setting_for_image_param(self):
        self.org_height = 1280
        self.org_width = 1920
        self.height = 384
        self.width = 640
        self.size = [self.width, self.height, self.width, self.height]
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.crop_size = 480
        self.scale_factor = dict()
        self.scale_factor['img'] = [8, 16, 32]
        self.scale_factor['seg'] = [8]


    def setting_for_dataloader(self):
        self.num_workers = 4
        self.data_flip = True

        self.mode_transform = 'custom'  # ['custom', 'basic', 'complex']

        self.sampling = False
        self.sampling_step = 3
        self.sampling_mode = 'image'  # ['video', 'image']

        self.batch_size = {'img': 2}

        self.lane_width = dict()
        self.lane_width['org'] = 5
        self.lane_width['seg'] = 1
        self.lane_width['mode'] = 'dilation'  # ['dilation', 'gaussian']
        self.lane_width['sigmaX'] = 0.07
        self.lane_width['sigmaY'] = 0.07
        self.lane_width['kernel'] = (3, 3)
        self.lane_width['iteration'] = 1

    def setting_for_lane_representation(self):
        self.min_y_coord = 0
        self.max_y_coord = 270
        self.node_num = self.max_y_coord
        self.py_coord = self.height - np.float32(np.round(np.linspace(self.max_y_coord, self.min_y_coord + 1, self.node_num)))
        self.py_coord_org = np.copy(self.py_coord)

        self.mode_interp = 'splrep'  # ['splrep', 'spline', 'linear', 'slinear']

    def setting_for_svd(self):
        self.top_m = 6

        # sampling lane component
        self.node_num = 100
        self.sample_idx = np.int32(np.linspace(0, self.max_y_coord - 1, self.node_num))
        self.node_sampling = True
        if self.node_sampling == True:
            self.py_coord = self.py_coord[self.sample_idx]

    def setting_for_visualization(self):
        self.disp_step = 50
        self.disp_test_result = False

    def setting_for_save(self):
        self.save_pickle = True

    def setting_for_training(self):
        self.run_mode = 'train'  # ['train', 'test', 'eval', 'val_for_training_set']
        self.resume = True

        self.epochs = 30

        self.optim = dict()
        self.optim['lr'] = 1e-4
        # self.optim['milestones'] = list(np.arange(0, 1e4 // self.batch_size['img'] * 30, 1e4 // self.batch_size['img']))[1:]
        self.optim['weight_decay'] = 1e-4
        self.optim['gamma'] = 0.5
        self.optim['betas'] = (0.9, 0.999)
        self.optim['eps'] = 1e-8
        self.optim['mode'] = 'adam_w'  # ['adam_w', 'adam']

        self.backbone = 'resnet18'

        self.iteration = dict()

    def setting_for_postprocessing(self):
        self.max_lane_num = 6

        self.pad = dict()
        self.pad['st'] = [5, 5]  # H W
        self.pad['ed'] = [10, 12, 13]
        self.nms_thresd = 0.5
        self.prob_thresd = 0.5
        self.height_thresd = 0.5
        self.removal = dict()
        self.removal['lane_width'] = 3
        self.removal_length = dict()
        self.removal_length['st'] = 3
        self.removal_length['ed'] = 3

    def setting_for_evaluation(self):
        self.param_name = 'trained_last'  # ['trained_last', 'max', 'min']

        self.do_eval_iou = False
        self.do_eval_iou_laneatt = True

        self.eval_h = self.org_height // 2
        self.eval_w = self.org_width // 2


        self.iou_thresd = dict()
        self.iou_thresd['laneatt'] = 0.5

    def setting_for_modeling(self):
        self.num_layers = dict()
        self.num_layers['enc'] = 3
        self.num_layers['dec'] = 3

        self.dropout = 0.0
        self.num_head = 1

        self.kernel_size = 3
        self.c_dims = 128
        self.c_dims2 = 128

        self.num_query = 82
        self.num_query_selected = 20