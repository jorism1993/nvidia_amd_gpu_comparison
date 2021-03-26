import torch
import torchvision.models as tv_models
from efficientnet_pytorch import EfficientNet


def move_to(var, device):
    if var is None:
        return None
    elif isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    elif isinstance(var, list):
        return [move_to(k, device) for k in var]
    elif isinstance(var, tuple):
        return (move_to(k, device) for k in var)
    return var.to(device)


def get_model(opts):
    if 'resnet' in opts.model_type or 'resnext' in opts.model_type:
        model_func = getattr(tv_models, opts.model_type)
        model = model_func()
    elif opts.model_type == 'inception_v3':
        model = tv_models.inception_v3(aux_logits=False)
    elif 'efficientnet' in opts.model_type:
        model = EfficientNet.from_pretrained(opts.model_type, image_size=opts.img_size)
    else:
        raise NotImplementedError('Invalid model type')

    return model
