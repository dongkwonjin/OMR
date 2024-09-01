import torch

def batch_to_cuda(batch, is_training=False):
    if is_training == True:
        namelist = list(batch)
    else:
        namelist = ['img']

    for name in namelist:
        if torch.is_tensor(batch[name]):
            batch[name] = batch[name].cuda()
        elif type(batch[name]) is dict:
            for key in batch[name].keys():
                batch[name][key] = batch[name][key].cuda()
    return batch

def finetune_model(model):
    val1 = False
    val2 = False
    for param in model.classifier.parameters():
        param.requires_grad = val1
    for param in model.feat_embedding.parameters():
        param.requires_grad = val2
    for param in model.regressor.parameters():
        param.requires_grad = val2
    for param in model.offset_regression.parameters():
        param.requires_grad = val2
    for param in model.mask_regression.parameters():
        param.requires_grad = val2
    for param in model.deform_conv2d.parameters():
        param.requires_grad = val2

    model.classifier.eval()
    model.feat_embedding.eval()
    model.regressor.eval()
    model.offset_regression.eval()
    model.mask_regression.eval()
    model.deform_conv2d.eval()

    return model

