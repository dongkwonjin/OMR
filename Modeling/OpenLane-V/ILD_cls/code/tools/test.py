import numpy as np

from libs.utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.testloader = dict_DB['testloader']
        self.query_selector = dict_DB['query_selector']
        self.post_process = dict_DB['post_process']
        self.save_pred_for_eval_iou = dict_DB['save_pred_for_eval_iou']
        self.eval_iou = dict_DB['eval_iou_official']
        self.eval_iou_laneatt = dict_DB['eval_iou_laneatt']
        self.eval_seg = dict_DB['eval_seg']
        self.visualizer = dict_DB['visualizer']

        self.mask = np.ones((self.cfg.height // self.cfg.scale_factor['seg'][0], self.cfg.width // self.cfg.scale_factor['seg'][0]), dtype=np.int32)

    def init_data(self):
        self.result = {'out': {}, 'gt': {}, 'name': []}
        self.datalist = []
        self.eval_seg.init()

    def batch_to_cuda(self, batch):
        name = 'img'
        if torch.is_tensor(batch[name]):
            batch[name] = batch[name].cuda()
        elif type(batch[name]) is dict:
            for key in batch[name].keys():
                batch[name][key] = batch[name][key].cuda()
        return batch

    # def determine_height_center(self, prob_map):
    #     batch_pad = list()
    #     b, _, h, w = prob_map.shape
    #     h_map = np.sum((to_np(prob_map[:, 0]) > self.cfg.height_thresd), axis=2)
    #     for i in range(b):
    #         idxlist = h_map[i].nonzero()[0]
    #         if len(idxlist) == 0:
    #             pad = self.cfg.pad['ed'][:2]
    #         else:
    #             pad = [h - int(np.median(idxlist)), self.cfg.pad["ed"][1]]
    #
    #         l = self.cfg.pad["ed"][2]
    #         b_left = self.mask[-pad[0] - l:-pad[0], pad[1]]
    #         b_bottom = self.mask[-pad[0], pad[1]:-pad[1]]
    #         b_right = self.mask[-pad[0] - l:-pad[0], -pad[1] - 1]
    #         if sum(b_left) + sum(b_bottom) + sum(b_right) != self.cfg.num_query:
    #             pad = self.cfg.pad['ed'][:2]
    #
    #         batch_pad.append(pad)
    #     batch_pad = np.array(batch_pad).transpose(1, 0)
    #     return {"boundary_pad": batch_pad}

    def run(self, model_occ, model, mode='val'):
        self.init_data()

        with torch.no_grad():
            model_occ.eval()
            model.eval()
            for i, batch in enumerate(self.testloader):  # load batch data
                batch = self.batch_to_cuda(batch)

                # model
                out = dict()

                model.forward_for_feature_extraction(batch['img'])
                model.img_feat = model.img_feat
                out.update(model.forward_for_classification())
                out.update(model.forward_for_classification_occ())
                # out.update(model.forward_for_regression())
                out.update(model_occ.forward_for_occlusion_detection(batch['img']))

                # out_post = self.post_process.run_for_nms(out)

                self.eval_seg.update(batch, out, mode)
                self.eval_seg.run_for_fscore()

                for j in range(len(batch['img'])):
                    # visualizer
                    if self.cfg.disp_test_result == True:
                        # out.update(out_post[j])
                        self.visualizer.display_for_test(batch=batch, out=out, batch_idx=j, mode=mode)

                    # # record output data
                    # self.result['out']['x_coords'] = out_post[j]['x_coords']
                    # self.result['out']['height_idx'] = out_post[j]['height_idx']
                    # self.result['name'] = batch['img_name'][j]
                    #
                    # if self.cfg.save_pickle == True:
                    #     save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/{batch["img_name"][j].replace(".jpg", "")}', data=self.result)

                self.datalist += batch['img_name']

                if i % 50 == 1:
                    print(f'image {i} ---> {batch["img_name"][0]} done!')

        if self.cfg.save_pickle == True:
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/datalist', data=self.datalist)
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/eval_seg_results', data=self.eval_seg.results)

        # evaluation
        return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()

        try:
            metric.update(self.eval_seg.measure())
        except:
            print('no seg metric')

        # if self.cfg.do_eval_iou_laneatt == True:
        #     self.save_pred_for_eval_iou.settings(key=['x_coords'], test_mode=mode, use_height=True)
        #     self.save_pred_for_eval_iou.run()
        #     metric.update(self.eval_iou_laneatt.measure_IoU(mode, self.cfg.iou_thresd['laneatt']))

        return metric