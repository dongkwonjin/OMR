import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg

    def get_video_datalist_for_processing(self):
        print('start')
        datalist_video = load_pickle(f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_video_{self.cfg.datalist_mode}')

        datalist_out = dict()
        video_list = list(datalist_video)
        num = 0

        num_frames = self.cfg.clip_length

        for i in range(len(video_list)):
            video_name = video_list[i]
            file_list = datalist_video[video_name]
            for j in range(len(file_list)):
                name = file_list[j]
                dirname = os.path.dirname(name)
                filename = os.path.basename(name)

                datalist_out[name] = dict()
                datalist_out[name]['current'] = list()
                datalist_out[name]['past'] = list()
                datalist_out[name]['future'] = list()
                for t in range(0, num_frames):
                    if j - t < 0:
                        break
                    prev_filename = file_list[j-t]
                    mode = 'current' if t == 0 else 'past'
                    datalist_out[name][mode].append(prev_filename)

                if len(datalist_out[name]['past']) != num_frames - 1 and self.cfg.datalist_mode == 'training':
                    datalist_out.pop(name)
                else:
                    if j + 1 < len(file_list):
                        datalist_out[name]['future'].append(file_list[j + 1])
                print(f'{num} ==> {filename} done')
                num += 1

        print(f'The number of datalist_video: {len(datalist_out)}')
        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_{num_frames}', data=datalist_out)

    def run(self):
        print('start')
        self.get_video_datalist_for_processing()