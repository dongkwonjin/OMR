import torch
from models.model1 import Model as Model1
from models.model2 import Model as Model2

from models.model import Model
from models.loss import *

def load_model_for_test(cfg, dict_DB):
    dict_DB['model1'] = load_pretrained_model1(cfg)
    dict_DB['model2'] = load_pretrained_model2(cfg)
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_openlane-v_OMR')
    else:
        if cfg.param_name == 'trained_last':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        elif cfg.param_name == 'max':
            checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_max_F1_{cfg.dataset_name}_{cfg.backbone}')
    model = Model(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    dict_DB['model'] = model
    return dict_DB

def load_model_for_train(cfg, dict_DB):
    dict_DB['model1'] = load_pretrained_model1(cfg)
    dict_DB['model2'] = load_pretrained_model2(cfg)
    model = Model(cfg=cfg)
    model = load_for_finetuning_pretrained_model(cfg, model)
    model.cuda()

    if cfg.optim['mode'] == 'adam_w':
        optimizer = torch.optim.AdamW(params=model.parameters(),
                                      lr=cfg.optim['lr'],
                                      weight_decay=cfg.optim['weight_decay'],
                                      betas=cfg.optim['betas'], eps=cfg.optim['eps'])
    elif cfg.optim['mode'] == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(),
                                     lr=cfg.optim['lr'],
                                     weight_decay=cfg.optim['weight_decay'])

    cfg.optim['milestones'] = list(np.arange(0, len(dict_DB['trainloader']) * cfg.epochs, len(dict_DB['trainloader']) * cfg.clip_length // 2))[1:]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                     milestones=cfg.optim['milestones'],
                                                     gamma=cfg.optim['gamma'])

    if cfg.resume == False:
        checkpoint = torch.load(f'{cfg.dir["weight"]}/checkpoint_final')
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                         milestones=cfg.optim['milestones'],
                                                         gamma=cfg.optim['gamma'],
                                                         last_epoch=checkpoint['batch_iteration'])
        dict_DB['epoch'] = checkpoint['epoch']
        dict_DB['iteration'] = checkpoint['iteration']
        dict_DB['batch_iteration'] = checkpoint['batch_iteration']
        dict_DB['val_result'] = checkpoint['val_result']

    loss_fn = Loss_Function(cfg)

    dict_DB['model'] = model
    dict_DB['optimizer'] = optimizer
    dict_DB['scheduler'] = scheduler
    dict_DB['loss_fn'] = loss_fn

    return dict_DB

def load_pretrained_model1(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}_{cfg.backbone}')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_seg_fscore_openlane-v_ILD_cls')
    model = Model1(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model

def load_pretrained_model2(cfg):
    checkpoint = torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}_{cfg.backbone}')
    if cfg.run_mode == 'test_paper':
        checkpoint = torch.load(f'{cfg.dir["weight_paper"]}/checkpoint_max_F1_openlane-v_ILD_reg')
    model = Model2(cfg=cfg)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.cuda()
    return model


def load_for_finetuning_pretrained_model(cfg, model):
    checkpoint = dict()
    checkpoint.update(torch.load(f'{cfg.dir["pretrained_weight1"]}/checkpoint_max_seg_fscore_{cfg.dataset_name}_{cfg.backbone}')['model'])
    checkpoint.update(torch.load(f'{cfg.dir["pretrained_weight2"]}/checkpoint_max_F1_{cfg.dataset_name}_{cfg.backbone}')['model'])

    paramlist = list()
    namelist = ['classifier', 'feat_embedding', 'regressor', 'offset_regression', 'mask_regression', 'deform_conv2d']
    for param in list(checkpoint):
        for name in namelist:
            if 'classifier_occ' in param:
                continue
            if name in param:
                paramlist.append(param)

    for param in list(checkpoint):
        if param not in paramlist:
            del checkpoint[param]

    model.load_state_dict(checkpoint, strict=False)
    model.cuda()
    return model
