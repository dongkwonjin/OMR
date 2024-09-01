import cv2
import json

import torch
import torch.nn.functional as F

from libs.utils import *


class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.area_thresd = 15000

    def get_datalist(self):
        with open(f'{self.cfg.dir["dataset"]}/update_{self.mode}_2020.json') as f:
            datalist = json.load(f)
        for i in range(len(datalist['images'])):
            data = datalist['images'][i]
            image_id = data['id'] + self.id
            file_name = f'{self.mode}ing/image_2/{data["file_name"]}'
            self.results[image_id] = dict()
            self.results[image_id]['file_name'] = file_name
            self.results[image_id]['annotations'] = list()

        for i in range(len(datalist['annotations'])):
            print(f'load {i}')
            data = datalist['annotations'][i]
            image_id = data['image_id'] + self.id
            area = data['a_area']
            if len(data['a_segm']) > 0 and area > self.area_thresd:
                self.results[image_id]['annotations'].append(data)
            else:
                print('no objects')
        # for i in range(len(self.results)):
        for i in range(len(datalist['images'])):
            data = datalist['images'][i]
            image_id = data['id'] + self.id
            if len(self.results[image_id]['annotations']) == 0:
                self.results.pop(image_id)
    def run(self):
        print('start')
        self.results = dict()
        self.id = 0
        for mode in ['train', 'test']:
            self.mode = mode
            self.get_datalist()
            self.id = list(self.results)[-1] + 1

        save_pickle(path=f'{self.cfg.dir["out"]}/pickle/datalist_{self.area_thresd}', data=self.results)