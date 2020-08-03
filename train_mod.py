import json
import argparse
from easydict import EasyDict

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from models.sintel_raw import SintelRawFlow
from models.sintel_ar import SintelARFlow
from utils.torch_utils import init_seed


def get_model_class(name):
    if name == 'Sintel':
        return SintelRawFlow
    elif name == 'Sintel_AR':
        return SintelARFlow
    elif name == 'KITTI':
        raise NotImplementedError(name)
    elif name == 'KITTI_AR':
        raise NotImplementedError(name)
    else:
        raise NotImplementedError(name)


def main(args, cfg):
    init_seed(cfg.seed)
    logger = WandbLogger(project='arflow')
    model_class = get_model_class(cfg.trainer)
    model = model_class(cfg)
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger
    )
    trainer.fit(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-c', '--config', default='configs/sintel_ft.json')
    # parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-m', '--model', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    # if args.evaluate:
    #     cfg.train.update({
    #         'epochs': 1,
    #         'epoch_size': -1,
    #         'valid_size': 0,
    #         'workers': 1,
    #         'val_epoch_size': 1,
    #     })

    if args.model is not None:
        cfg.train.pretrained_model = args.model

    main(args, cfg)
