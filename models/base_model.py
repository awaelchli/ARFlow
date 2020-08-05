import torch

from datasets.get_dataset import get_dataset
from losses.get_loss import get_loss
from models.get_model import get_model
from utils.torch_utils import bias_parameters, weight_parameters, \
    load_checkpoint, AdamW
from pytorch_lightning import LightningModule


class ARFlow(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.train
        self.model = self._init_model(get_model(cfg.model))
        self.loss_func = get_loss(cfg.loss)
        self.train_set, self.valid_set = get_dataset(cfg)

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

    def test_dataloader(self):
        return self.val_dataloader()

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
