import cv2
import torch
import torch.nn.functional as F

from libs.utils import *

class Preprocessing(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.thresd_frame_interval = 6  # sample_num * interval

    def get_video_datalist(self, mode):
        if mode == 'org':
            datalist = load_pickle(f'{self.cfg.dir["pre0"]}/datalist')
        elif mode == 'filtered':
            path = f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_{self.cfg.datalist_mode}'
            # if self.cfg.datalist_mode == 'validation':
            #     path = f'{self.cfg.dir["dataset"]}/OpenLane-V/list/datalist_{self.cfg.datalist_mode}'
            # elif self.cfg.datalist_mode == 'training':
            #     path = f'{self.cfg.dir["pre2"]}/datalist'
            datalist = load_pickle(path)

        datalist_out = dict()
        for i in range(len(datalist)):
            name = datalist[i]
            dirname = os.path.dirname(name)
            filename = os.path.basename(name)

            if dirname not in datalist_out.keys():
                datalist_out[dirname] = list()
            datalist_out[dirname].append(name)

            print(f'{i} ==> {name} done')

        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video_{mode}', data=datalist_out)

    def matching_two_datalist(self):
        datalist_match = dict()
        datalist_filtered = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video_filtered')
        datalist = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video_org')

        dirname_list = list(datalist_filtered)
        for dirname in dirname_list:
            filename_list1 = datalist_filtered[dirname]
            filename_list2 = datalist[dirname]
            datalist_match[dirname] = list()
            for file_name in filename_list1:
                try:
                    match_idx = filename_list2.index(file_name)
                except:
                    print('err')
                datalist_match[dirname].append(match_idx)

        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_match', data=datalist_match)

    def split_video_datalist(self):
        datalist_video_out = dict()
        datalist_out = list()
        datalist_match = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_match')
        datalist_filtered = load_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_video_filtered')

        dirname_list = list(datalist_filtered)
        for dirname in dirname_list:
            filename_list = datalist_filtered[dirname]
            index_list = datalist_match[dirname]

            split_val = 0
            datalist_video_out[f'{dirname}_split{split_val}'] = list()

            for i in range(len(filename_list)):
                if i == 0:
                    datalist_video_out[f'{dirname}_split{split_val}'].append(filename_list[i])
                    datalist_out.append(filename_list[i])
                    continue
                if index_list[i] - index_list[i - 1] > self.thresd_frame_interval:
                    split_val += 1
                    datalist_video_out[f'{dirname}_split{split_val}'] = list()

                datalist_video_out[f'{dirname}_split{split_val}'].append(filename_list[i])
                datalist_out.append(filename_list[i])
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_split_video', data=datalist_video_out)
        save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_split', data=datalist_out)


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
                for t in range(0, num_frames):
                    if j - t < 0:
                        break
                    prev_filename = file_list[j-t]
                    datalist_out[name][f"t-{t}"] = prev_filename
                    if len(datalist_out[name]) == self.cfg.clip_length:
                        break
                if len(datalist_out[name]) != self.cfg.clip_length and self.cfg.datalist_mode == 'training':
                    datalist_out.pop(name)
                else:
                    if j + 1 < len(file_list):
                        datalist_out[name][f"t+1"] = file_list[j + 1]
                print(f'{num} ==> {filename} done')
                num += 1

        print(f'The number of datalist_video: {len(datalist_out)}')
        if self.cfg.save_pickle == True:
            save_pickle(f'{self.cfg.dir["out"]}/pickle/datalist_{self.cfg.clip_length}', data=datalist_out)

    def run(self):
        print('start')

        # self.get_video_datalist(mode='org')
        # self.get_video_datalist(mode='filtered')
        # self.matching_two_datalist()
        # self.split_video_datalist()
        self.get_video_datalist_for_processing()