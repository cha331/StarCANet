import os
import random

import numpy as np
import thop
import torch

from models import starcanet
from utils import logger, line_seg

__all__ = ["init_device", "init_model"]


def init_device(seed=None, cpu=None, gpu=None, affinity=None):
    # set the CPU affinity
    if affinity is not None:
        os.system(f'taskset -p {affinity} {os.getpid()}')

    # Set the random seed
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Set the GPU id you choose
    if gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    # Env setup
    if not cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        if seed is not None:
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
        pin_memory = True
        logger.info("Running on GPU%d" % (gpu if gpu else 0))
    else:
        pin_memory = False
        device = torch.device('cpu')
        logger.info("Running on CPU")

    return device, pin_memory


def init_model(cr=4, size='L',pretrained=None):
    # Model loading
    model = starcanet(reduction=cr, model_size=size)

    if pretrained is not None:
        assert os.path.isfile(pretrained)
        state_dict = torch.load(pretrained, map_location=torch.device('cpu'), weights_only=False)['state_dict']
        model.load_state_dict(state_dict, strict=False)
        logger.info("pretrained model loaded from {}".format(pretrained))

    # from ptflops import get_model_complexity_info
    #
    # flops, params = get_model_complexity_info(model, (2, 32, 32), as_strings=True, print_per_layer_stat=True)
    # logger.info(f'=> Model Flops: {flops}')
    # logger.info(f'=> Model Params Num: {params}\n')
    # Model flops and params counting
    image = torch.randn([1, 2, 32, 32])
    flops, params = thop.profile(model, inputs=(image,), verbose=False)
    flops, params = thop.clever_format([flops, params], "%.3f")

    # Model info logging
    logger.info(f'=> Model Name: StarCANet-{size} [pretrained: {pretrained}]')
    logger.info(f'=> Model Config: compression ratio=1/{cr}')
    logger.info(f'=> Model Flops: {flops}')
    logger.info(f'=> Model Params Num: {params}\n')
    logger.info(f'{line_seg}\n{model}\n{line_seg}\n')

    return model
