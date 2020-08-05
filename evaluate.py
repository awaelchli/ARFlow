import json
import argparse
import torch
import wandb
from easydict import EasyDict

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models.sintel_raw import SintelRawFlow
from models.sintel_ar import SintelARFlow
from utils.flow_utils import evaluate_flow
from utils.flow_utils import flow_to_image
from mlutils.optical_flow import flow2rgb


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
    device = torch.device('cuda', 0)
    logger = WandbLogger(project='arflow')
    model_class = get_model_class(cfg.trainer)
    model = model_class(cfg)
    model = model.to(device)

    model.eval()
    dl = model.test_dataloader()
    if isinstance(dl, list):
        dl = dl[1]  # final

    epes = []
    for i, batch in enumerate(dl):
        img1, img2 = batch['img1'], batch['img2']
        img_pair = torch.cat([img1, img2], 1).to(device)
        gt_flows = batch['target']['flow'].cpu()

        # compute output
        flows = model(img_pair)['flows_fw']
        pred_flows = flows[0].detach().cpu()
        es = evaluate_flow(gt_flows.numpy().transpose([0, 2, 3, 1]), pred_flows.numpy().transpose([0, 2, 3, 1]))
        epe = torch.tensor(es).cpu()
        epes.append(epe)

        if i % 10 == 0:
            logger.experiment.log({
                'flow': [wandb.Image(flow2rgb(torch.as_tensor(pred_flows)))],
                'gt': [wandb.Image(flow2rgb(torch.as_tensor(gt_flows)))]
            })

    logger.experiment.log({
        'epe': torch.stack(epes).mean().item()
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('-c', '--config', default='configs/sintel_ft.json')
    parser.add_argument('-m', '--model', default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = EasyDict(json.load(f))

    if args.model is not None:
        cfg.train.pretrained_model = args.model

    # checkpoints
    # trained sintel_raw
    "arflow/3b6q1t3d/checkpoints/epoch=41.ckpt"
    # trained sintel_ft_ar
    "arflow/u04khjlv/checkpoints/epoch=352.ckpt"

    main(args, cfg)