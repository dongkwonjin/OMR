import torch

from libs.save_model import *
from libs.utils import *
from libs.train_utils import *

class Train_Process(object):
    def __init__(self, cfg, dict_DB):
        self.cfg = cfg
        self.sf = self.cfg.scale_factor['seg'][0]

        self.dataloader = dict_DB['trainloader']

        self.model = dict_DB['model']
        self.model1 = dict_DB['model1']
        self.model2 = dict_DB['model2']
        # self.model3 = dict_DB['model3']

        self.optimizer = dict_DB['optimizer']
        self.scheduler = dict_DB['scheduler']
        self.loss_fn = dict_DB['loss_fn']
        self.visualizer = dict_DB['visualizer']
        self.post_processing = dict_DB['post_processing2']

        self.test_process = dict_DB['test_process']
        self.val_result = dict_DB['val_result']

        self.logfile = dict_DB['logfile']
        self.epoch_s = dict_DB['epoch']
        self.iteration = dict_DB['iteration']
        self.batch_iteration = dict_DB['batch_iteration']

        self.cfg.iteration['validation'] = len(self.dataloader) // 1
        self.cfg.iteration['record'] = self.cfg.iteration['validation'] // 4
        self.vm = dict_DB['video_memory']

    def training(self):
        loss_t = dict()
        clip_length = self.cfg.clip_length
        # train start
        self.model1.eval()
        self.model2.eval()
        self.model.train()
        self.model = finetune_model(self.model)
        logger('train start\n', self.logfile)
        for i, batch in enumerate(self.dataloader):
            # load data
            batch = batch_to_cuda(batch, is_training=True)
            batch_t = dict()
            B, H, W = batch['seg_label'][self.sf].shape
            batch_t['seg_label'] = batch['seg_label'][self.sf].view(-1, clip_length, H, W)
            batch_t['coeff_label'] = batch['coeff_label'][self.sf].view(-1, clip_length, H, W, self.cfg.top_m)
            batch_t['pad'] = torch.cat(batch['pad']).view(2, -1).view(2, -1, clip_length)

            # model
            out = dict()
            self.model1.forward_for_feature_extraction(batch['img'])
            out.update(self.model1.forward_for_classification())
            out.update(self.model1.forward_for_classification_occ())
            out.update(self.model2.forward_for_regression(self.model1.prob_map))

            B, C, H, W = self.model1.img_feat.shape
            self.model1.img_feat = self.model1.img_feat.view(-1, clip_length, C, H, W)
            self.model1.obj_mask = self.model1.obj_mask.view(-1, clip_length, 1, H, W)

            loss_per_clip = dict()
            for j in range(clip_length):
                key_t = f't-0'
                self.vm.key_t = key_t
                self.vm.prev_frame_num = j
                self.vm.forward_for_dict_preparation()
                self.vm.forward_for_dict_update_per_frame(self.model1, t_idx=j, mode='intra', is_training=True)
                self.model = self.vm.forward_for_dict_transfer(self.model)

                out[j] = dict()

                out[j].update(self.model.forward_for_data_translation(is_training=True))
                if self.vm.memory_t > 1:
                    # self.model.forward_for_feature_refinement(is_training=True)
                    out[j].update(self.model.forward_for_classification(self.model.curr_img_feat_lstm))
                    out[j].update(self.model.forward_for_regression(self.model.prob_map))
                    self.vm.forward_for_dict_update_per_frame(self.model, mode='update')
                else:
                    out[j]['prob_map'] = out['prob_map_init'].view(-1, clip_length, 1, H, W)[:, 0]
                    out[j]['coeff_map'] = out['coeff_map_init'].view(-1, clip_length, self.cfg.top_m, H, W)[:, 0]

                # loss
                loss_per_clip[j] = self.loss_fn(out=out[j], gt=batch_t, t=j)

                # post processing
                self.post_processing.pad = batch_t['pad'][:, :, j]
                out[j].update(self.post_processing.run_for_nms(out[j]))
                out_post = self.post_processing.run(out[j])
                # out[j].update(out_post)

                # lane mask generation
                self.post_processing.seg_label = batch_t['seg_label'][:, j]
                self.post_processing.coeff_label = batch_t['coeff_label'][:, j]
                out[j].update(self.post_processing.run_for_lane_mask_generation(out_post, prev_frame_num=self.vm.prev_frame_num, is_training=True))
                self.vm.data['lane_mask']['t-0'] = out[j]['lane_mask'].clone()
                self.vm.data['lane_pos_map']['t-0'] = out[j]['lane_pos_map'].clone()

            # visualization
            if i % self.cfg.disp_step == 0:
                self.visualizer.display_for_training(batch, out, batch_idx=i, t_idx=j)

            # total loss for a video clip
            loss = dict()
            for t_idx in loss_per_clip:
                for key_name in loss_per_clip[t_idx].keys():
                    if key_name not in loss.keys():
                        loss[key_name] = torch.FloatTensor([0.0]).cuda()
                    loss[key_name] += loss_per_clip[t_idx][key_name]
            for key_name in loss.keys():
                loss[key_name] /= len(loss_per_clip)

            # optimize
            self.optimizer.zero_grad()
            loss['sum'].backward()
            self.optimizer.step()

            for l_name in loss:
                if l_name not in loss_t.keys():
                    loss_t[l_name] = 0
                loss_t[l_name] += loss[l_name].item()

            if i % self.cfg.disp_step == 0:
                logger(f'epoch : {self.epoch}, batch_iteration {self.batch_iteration}, iteration : {self.iteration}, iter per epoch : {i} ==> {batch["img_name"][0]} ', self.logfile, option='space')
                for l_name in loss:
                    logger(f'Loss_{l_name} : {round(loss[l_name].item(), 4)}, ', self.logfile, option='space')
                logger('\n', self.logfile)

            self.iteration += self.cfg.batch_size['img']
            self.batch_iteration += 1

            if (self.batch_iteration % self.cfg.iteration['record']) == 0 or (self.batch_iteration % self.cfg.iteration['validation']) == 0:
                # save model
                self.ckpt = {'epoch': self.epoch,
                             'iteration': self.iteration,
                             'batch_iteration': self.batch_iteration,
                             'model': self.model,
                             'optimizer': self.optimizer,
                             'val_result': self.val_result}

                save_model(checkpoint=self.ckpt, param='checkpoint_final', path=self.cfg.dir['weight'])

            if (self.batch_iteration % self.cfg.iteration['record']) == 0:
                # logger
                logger('\nAverage Loss : ', self.logfile, option='space')
                for l_name in loss_t:
                    logger(f'{l_name} : {round(loss_t[l_name] / (i + 1), 6)}, ', self.logfile, option='space')
                logger('\n', self.logfile)

            if (self.batch_iteration % self.cfg.iteration['validation'] == 0) and (self.epoch + 1) % 5 == 0:
                self.test()
                self.model.train()
                self.model = finetune_model(self.model)

            self.scheduler.step(self.batch_iteration)

    def test(self):
        metric = self.test_process.run(self.model1, self.model2, self.model, mode='val')

        logger(f'\nEpoch {self.ckpt["epoch"]} Iteration {self.ckpt["iteration"]} ==> Validation result', self.logfile)
        for key in metric.keys():
            logger(f'{key} {metric[key]}\n', self.logfile)

        namelist = ['F1']
        for name in namelist:
            model_name = f'checkpoint_max_{name}_{self.cfg.dataset_name}_{self.cfg.backbone}'
            self.val_result[name] = save_model_max(self.ckpt, self.cfg.dir['weight'], self.val_result[name], metric[name], logger, self.logfile, model_name)

    def run(self):
        for epoch in range(self.epoch_s, self.cfg.epochs):
            self.epoch = epoch

            logger(f'\nepoch {epoch}\n', self.logfile)

            self.dataloader.dataset.generate_datalist()
            self.training()
