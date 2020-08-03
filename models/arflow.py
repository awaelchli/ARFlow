from copy import deepcopy

import torch

from datasets.get_dataset import get_dataset
from losses.get_loss import get_loss
from models.get_model import get_model
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from utils.flow_utils import evaluate_flow
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, AdamW
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop
from pytorch_lightning import LightningModule


class ARFlow(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.train
        self.model = self._init_model(get_model(cfg.model))
        self.loss_func = get_loss(cfg.loss)
        self.train_set, self.valid_set = get_dataset(cfg)
        self.sp_transform = RandomAffineFlow(self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise)

    def train_dataloader(self):
        cfg = self.cfg
        train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=cfg.batch_size,
            num_workers=cfg.workers, pin_memory=True, shuffle=True)

        if cfg.epoch_size == 0:
            cfg.epoch_size = len(train_loader)

        cfg.epoch_size = min(cfg.epoch_size, len(train_loader))
        return train_loader

    def val_dataloader(self):
        cfg = self.cfg
        max_test_batch = 4
        if type(self.valid_set) is torch.utils.data.ConcatDataset:
            valid_loader = [torch.utils.data.DataLoader(
                s, batch_size=min(max_test_batch, cfg.batch_size),
                num_workers=min(4, cfg.workers),
                pin_memory=True, shuffle=False) for s in self.valid_set.datasets]
            valid_size = sum([len(l) for l in valid_loader])
        else:
            valid_loader = torch.utils.data.DataLoader(
                self.valid_set, batch_size=min(max_test_batch, cfg.batch_size),
                num_workers=min(4, cfg.workers),
                pin_memory=True, shuffle=False)
            valid_size = len(valid_loader)

        if cfg.valid_size == 0:
            cfg.valid_size = valid_size

        cfg.valid_size = min(cfg.valid_size, valid_size)
        return valid_loader

    def _init_model(self, model):
        if self.cfg.pretrained_model:
            epoch, weights = load_checkpoint(self.cfg.pretrained_model)

            from collections import OrderedDict
            new_weights = OrderedDict()
            model_keys = list(model.state_dict().keys())
            weight_keys = list(weights.keys())
            for a, b in zip(model_keys, weight_keys):
                new_weights[a] = weights[b]
            weights = new_weights
            model.load_state_dict(weights)
        else:
            model.init_weights()
        return model

    def configure_optimizers(self):
        param_groups = [
            {'params': bias_parameters(self.model),
             'weight_decay': self.cfg.bias_decay},
            {'params': weight_parameters(self.model),
             'weight_decay': self.cfg.weight_decay}]

        if self.cfg.optim == 'adamw':
            optimizer = AdamW(param_groups, self.cfg.lr,
                              betas=(self.cfg.momentum, self.cfg.beta))
        elif self.cfg.optim == 'adam':
            optimizer = torch.optim.Adam(param_groups, self.cfg.lr,
                                         betas=(self.cfg.momentum, self.cfg.beta),
                                         eps=1e-7)
        else:
            raise NotImplementedError(self.cfg.optim)
        return optimizer

    def on_epoch_start(self):
        if 'stage1' in self.cfg:
            if self.current_epoch == self.cfg.stage1.epoch:
                self.loss_func.cfg.update(self.cfg.stage1.loss)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        img1, img2 = batch['img1'], batch['img2']
        img_pair = torch.cat([img1, img2], 1)

        # run 1st pass
        res_dict = self.model(img_pair, with_bk=True)
        flows_12, flows_21 = res_dict['flows_fw'], res_dict['flows_bw']
        flows = [torch.cat([flo12, flo21], 1) for flo12, flo21 in
                 zip(flows_12, flows_21)]
        loss, l_ph, l_sm, flow_mean = self.loss_func(flows, img_pair)

        flow_ori = res_dict['flows_fw'][0].detach()

        if self.cfg.run_atst:
            img1, img2 = batch['img1_ph'], batch['img2_ph']

            # construct augment sample
            noc_ori = self.loss_func.pyramid_occu_mask1[0]  # non-occluded region
            s = {'imgs': [img1, img2], 'flows_f': [flow_ori], 'masks_f': [noc_ori]}
            st_res = self.sp_transform(deepcopy(s)) if self.cfg.run_st else deepcopy(s)
            flow_t, noc_t = st_res['flows_f'][0], st_res['masks_f'][0]

            # run 2nd pass
            img_pair = torch.cat(st_res['imgs'], 1)
            flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]

            if not self.cfg.mask_st:
                noc_t = torch.ones_like(noc_t)
            l_atst = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
            l_atst = (l_atst * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += self.cfg.w_ar * l_atst
        else:
            l_atst = torch.zeros_like(loss)

        if self.cfg.run_ot:
            img1, img2 = batch['img1_ph'], batch['img2_ph']
            # run 3rd pass
            img_pair = torch.cat([img1, img2], 1)

            # random crop images
            img_pair, flow_t, occ_t = random_crop(img_pair, flow_ori, 1 - noc_ori, self.cfg.ot_size)

            # slic 200, random select 8~16
            if self.cfg.ot_slic:
                img2 = img_pair[:, 3:]
                seg_mask = run_slic_pt(img2, n_seg=200,
                                       compact=self.cfg.ot_compact, rd_select=[8, 16],
                                       fast=self.cfg.ot_fast).type_as(img2)  # Nx1xHxW
                noise = torch.rand(img2.size()).type_as(img2)
                img2 = img2 * (1 - seg_mask) + noise * seg_mask
                img_pair[:, 3:] = img2

            flow_t_pred = self.model(img_pair, with_bk=False)['flows_fw'][0]
            noc_t = 1 - occ_t
            l_ot = ((flow_t_pred - flow_t).abs() + self.cfg.ar_eps) ** self.cfg.ar_q
            l_ot = (l_ot * noc_t).mean() / (noc_t.mean() + 1e-7)

            loss += self.cfg.w_ar * l_ot
        else:
            l_ot = torch.zeros_like(loss)

        scaled_loss = 1024. * loss

        log_dict = {
            'loss': loss.item(),
            'l_ph': l_ph.item(),
            'l_sm': l_sm.item(),
            'flow_mean': flow_mean.item(),
            'l_atst': l_atst.item(),
            'l_ot': l_ot.item(),
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
