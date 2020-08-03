from copy import deepcopy

import torch

from models.base_model import ARFlow
from utils.flow_utils import evaluate_flow


class SintelRawFlow(ARFlow):

    def __init__(self, cfg):
        super().__init__(cfg)

    def training_step(self, batch, batch_idx):
        # read data to device
        img1, img2 = batch['img1'], batch['img2']
        img_pair = torch.cat([img1, img2], 1)
        # compute output
        res_dict = self.model(img_pair, with_bk=True)
        flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                 zip(flows_12, flows_21)]
        loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair)
        scaled_loss = 1024. * loss
        log_dict = {
            'loss': loss.item(),
            'l_ph': l_ph.item(),
            'l_sm': l_sm.item(),
            'flow_mean': flow_mean.item(),
        }
        return {
            'loss': scaled_loss,
            'log': log_dict,
        }

    def on_after_backward(self):
        for param in [p for p in self.model.parameters() if p.requires_grad]:
            param.grad.data.mul_(1. / 1024)

    def validation_step(self, batch, batch_idx, dataset_idx):
        img1, img2 = batch['img1'], batch['img2']
        img_pair = torch.cat([img1, img2], 1)
        gt_flows = batch['target']['flow'].cpu().numpy().transpose([0, 2, 3, 1])

        # compute output
        flows = self.model(img_pair)['flows_fw']
        pred_flows = flows[0].detach().cpu().numpy().transpose([0, 2, 3, 1])
        es = evaluate_flow(gt_flows, pred_flows)
        epe = torch.tensor(es).mean()
        return {'val_epe': epe}

    def validation_epoch_end(self, outputs):
        log_dict = {}
        for dataset_idx, out in enumerate(outputs):
            epe = torch.stack([output['val_epe'] for output in out]).mean()
            log_dict.update({f'val_epe_{dataset_idx}': epe})
        return {
            'val_loss': log_dict['val_epe_0'],
            'log': log_dict
        }
