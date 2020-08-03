from copy import deepcopy

import torch

from models.base_model import ARFlow
from transforms.ar_transforms.sp_transfroms import RandomAffineFlow
from utils.flow_utils import evaluate_flow
from transforms.ar_transforms.oc_transforms import run_slic_pt, random_crop


class SintelARFlow(ARFlow):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sp_transform = RandomAffineFlow(self.cfg.st_cfg, addnoise=self.cfg.st_cfg.add_noise)

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
