import numpy as np
from libs.utils import *
from libs.train_utils import *

class Test_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.sf = self.cfg.scale_factor['seg'][0]

        self.testloader = dict_DB['testloader']
        self.post_processing = dict_DB['post_processing']
        self.save_pred_for_eval_iou = dict_DB['save_pred_for_eval_iou']
        self.eval_iou_laneatt = dict_DB['eval_iou_laneatt']
        self.eval_temporal = dict_DB['eval_temporal']
        self.eval_seg = dict_DB['eval_seg']
        self.eval_flow = dict_DB['eval_flow']
        self.visualizer = dict_DB['visualizer']

        self.vm = dict_DB['video_memory']

    def init_data(self):
        self.result = {'out': {}, 'gt': {}, 'name': []}
        self.datalist = []
        self.eval_seg.init()
        self.eval_flow.init()

    def run(self, model1, model2, model, mode='val'):
        self.init_data()

        # self.visualizer.make_video(mode)
        with torch.no_grad():
            model1.eval()
            model2.eval()
            model.eval()
            for i, batch in enumerate(self.testloader):  # load batch data
                batch = batch_to_cuda(batch)

                # model
                out = dict()
                model1.forward_for_feature_extraction(batch['img'])
                out.update(model1.forward_for_classification())
                out.update(model1.forward_for_classification_occ())
                out.update(model2.forward_for_regression(model1.prob_map))

                for j in range(len(batch['prev_num'])):
                    prev_frame_num = int(batch['prev_num'][j])
                    self.vm.key_t = f't-0'
                    self.vm.prev_frame_num = prev_frame_num
                    self.vm.forward_for_dict_preparation()
                    self.vm.forward_for_dict_update_per_frame(model1, t_idx=j, mode='intra', is_training=False)
                    model = self.vm.forward_for_dict_transfer(model)

                    # feature refinement
                    out.update(model.forward_for_data_translation(is_training=False))
                    if self.vm.memory_t > 1:
                        out.update(model.forward_for_classification(model.curr_img_feat_lstm))
                        out.update(model.forward_for_regression(model.prob_map))
                        self.vm.forward_for_dict_update_per_frame(model, mode='update')
                    else:
                        out['prob_map'] = out['prob_map_init'][j:j+1]
                        out['coeff_map'] = out['coeff_map_init'][j:j+1]

                    # post processing
                    out_post = self.post_processing.run_for_nms(out)
                    out.update(out_post[0])

                    # lane mask generation
                    out.update(self.post_processing.run_for_lane_mask_generation(out_post))
                    self.vm.data['lane_mask']['t-0'] = out['lane_mask'].clone()
                    self.vm.data['lane_pos_map']['t-0'] = out['lane_pos_map'].clone()

                    # visualizer
                    if self.cfg.disp_test_result == True:
                        self.visualizer.prev_frame_num = prev_frame_num
                        self.visualizer.display_for_test(batch=batch, out=out, batch_idx=j, mode=mode)

                    self.eval_seg.update(batch, out, mode)
                    self.eval_seg.run_for_fscore()

                    # record output data
                    self.result['out']['x_coords'] = out['x_coords']
                    self.result['out']['height_idx'] = out['height_idx']
                    self.result['name'] = batch['img_name'][j]

                    if self.cfg.save_pickle == True:
                        save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/{batch["img_name"][j].replace(".jpg", "")}', data=self.result)

                    self.datalist.append(batch['img_name'][j])

                if i % 50 == 1:
                    print(f'image {i} ---> {batch["img_name"][0]} done!')

        if self.cfg.save_pickle == True:
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/datalist', data=self.datalist)
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/eval_seg_results', data=self.eval_seg.results)
            save_pickle(path=f'{self.cfg.dir["out"]}/{mode}/pickle/eval_flow_results', data=self.eval_flow.results)

        # evaluation
        return self.evaluation(mode)

    def evaluation(self, mode):
        metric = dict()

        try:
            metric.update(self.eval_seg.measure())
        except:
            print('no seg metric')
        try:
            metric.update(self.eval_flow.measure())
        except:
            print('no flow metric')

        if self.cfg.do_eval_iou_laneatt == True:
            self.save_pred_for_eval_iou.settings(key=['x_coords'], test_mode=mode, use_height=True)
            self.save_pred_for_eval_iou.run()
            metric.update(self.eval_iou_laneatt.measure_IoU(mode, self.cfg.iou_thresd['laneatt']))

        if self.cfg.do_eval_temporal == True:
            metric.update(self.eval_temporal.measure_IoU(mode, self.cfg.iou_thresd['temporal']))

        return metric