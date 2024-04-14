# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.optim.optimizer import Optimizer

from mysrc.utils.comm import create_dir
from mysrc.utils.ema import EMA
from mysrc.utils.logger import TxtLogger
from torch.utils.data import Dataset, DataLoader
import tqdm
from mysrc.utils.meter import AverageMeter

from torch.cuda.amp import autocast, GradScaler

from transformers import (get_cosine_schedule_with_warmup,
                          get_linear_schedule_with_warmup,
                          AdamW)

import torch.nn.functional as F
import torch.nn as nn


class My_Trainer():
    def __init__(self,
                 model: torch.nn.Module,
                 loss_fn,
                 optimizer: Optimizer,
                 scheduler,
                 logger: TxtLogger,
                 save_dir: str,
                 val_steps=1000,
                 log_steps=100,
                 device='cpu',
                 gradient_accum_steps=1,
                 max_grad_norm=3.0,
                 batch_to_model_inputs_fn=None,
                 early_stop_n=30,
                 fp16=False,
                 with_ema=False,
                 save_name_prefix='',
                 data_type='raw'):

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.val_steps = val_steps
        self.log_steps = log_steps
        self.logger = logger
        self.device = device
        self.gradient_accum_steps = gradient_accum_steps
        self.max_grad_norm = max_grad_norm
        self.batch_to_model_inputs_fn = batch_to_model_inputs_fn
        self.early_stop_n = early_stop_n
        self.global_step = 0
        self.fp16 = fp16

        self.with_ema = with_ema
        if with_ema:
            print('with ema!')
            self.ema = EMA(self.model, 0.99)
            self.ema.register()
        self.save_name_prefix = save_name_prefix

        if self.fp16:
            self.scaler = GradScaler()
        self.data_type = data_type#'raw'#'hybrid'

    def step(self, epo, step_n, batch):
        self.model.zero_grad()
        if self.data_type == 'hybrid':
            x1, x2, y = batch
            x1 = x1.cuda()
            x2 = x2.cuda()
            y = y.cuda()
        else:
            x, y = batch
            x = x.cuda()
            y = y.cuda()
        with autocast():
            if self.data_type == 'hybrid':
                pred, y = self.model(x1, x2, y)
            else:
                pred, y = self.model(x, y)
            #pred = self.model(x)
            loss = self.loss_fn(pred, y)

        if self.gradient_accum_steps > 1:
            loss = loss / self.gradient_accum_steps
        if self.fp16:
            if not torch.isnan(loss):  # Check if loss is not NaN
                self.scaler.scale(loss).backward()
            else:
                print('[WARN] loss is nan, skip backward')
        else:
            if not torch.isnan(loss):  # Check if loss is not NaN
                loss.backward()

        if (step_n + 1) % self.gradient_accum_steps == 0:

            # if self.fp16:
            #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)
            # else:
            #     torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # 调用 step 来更新参数
            self.scaler.step(self.optimizer)
            # 调用 update 来更新scaler的状态
            self.scaler.update()

            # self.optimizer.step()
            # if self.scheduler is not None:
            #     self.scheduler.step()

            if self.with_ema:
                self.ema.update()

        self.global_step += 1
        return loss

    def load_best(self):
        model_path = self.save_dir + "/{}best.pth".format(self.save_name_prefix)
        #print('load from: ', model_path)
        self.model.load_state_dict(torch.load(model_path))

    def predict(self, val_dataloader: DataLoader, return_loss=False):
        print("predict...")
        self.model.eval()
        preds = []
        if return_loss:
            eval_loss = 0.0
        for batch in tqdm.tqdm(val_dataloader):
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.data_type == 'hybrid':
                    x1, x2, y = batch
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    pred = self.model(x1, x2)
                else:
                    x, y = batch
                    x = x.cuda()
                    pred = self.model(x)
                if return_loss:
                    y = y.cuda()
                    loss = self.loss_fn(pred, y)
                    eval_loss += loss.item()
                preds.append(pred.float().cpu().numpy())
        if return_loss:
            eval_loss = eval_loss / len(val_dataloader)
            return np.concatenate(preds), eval_loss
        return np.concatenate(preds)

    def val_loss(self, val_dataloader: DataLoader):
        print("eval...")
        eval_loss = 0.0
        self.model.eval()

        for batch in tqdm.tqdm(val_dataloader):
            with torch.no_grad(), torch.cuda.amp.autocast():
                if self.data_type == 'hybrid':
                    x1, x2, y = batch
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    pred = self.model(x1, x2)
                else:
                    x, y = batch
                    x = x.cuda()
                    pred = self.model(x)
                y = y.cuda()
                loss = self.loss_fn(pred, y)
                eval_loss += loss.item()

        eval_loss = eval_loss / len(val_dataloader)
        return {'eval_loss': eval_loss}

    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              epoches=100):

        best_loss = 1e5
        early_n = 0
        print(epoches)
        for epo in range(epoches):
            step_n = 0
            train_avg_loss = AverageMeter()

            train_data_iter = tqdm.tqdm(train_dataloader)
            for batch in train_data_iter:
                self.model.train()

                train_loss = self.step(epo, step_n, batch)
                train_avg_loss.update(train_loss.item(), 1)

                status = '[{0}] lr={1:.7f} l={2:.5f} avg_l={3:.5f} '.format(
                    epo + 1, self.optimizer.param_groups[0]['lr'],
                    train_loss.item(), train_avg_loss.avg)

                train_data_iter.set_description(status)
                step_n += 1
                if self.global_step % self.val_steps == 0:

                    if self.with_ema:
                        self.ema.apply_shadow()
                    loss_dict = self.val_loss(val_dataloader)
                    eval_loss = loss_dict['eval_loss']

                    if best_loss >= eval_loss:
                        early_n = 0
                        best_loss = eval_loss
                        torch.save(self.model.state_dict(), self.save_dir + "/{}best.pth".format(self.save_name_prefix))

                    else:
                        early_n += 1
                    msg = "steps: {},epo: {}, best_loss: {:.6f} train_loss: {:.6f} ".format(
                        self.global_step, epo + 1, best_loss, train_avg_loss.avg)

                    for k, v in loss_dict.items():
                        msg += ', {}: {} '.format(k, v)

                    self.logger.write(msg)

                    if self.with_ema:
                        self.ema.restore()
            if epo != epoches - 1:
                self.scheduler.step()

            if self.with_ema:
                self.ema.apply_shadow()
            # torch.save(self.model.state_dict(),
            #            self.save_dir + "/{}latest.pth".format(self.save_name_prefix, epo + 1))
            if self.with_ema:
                self.ema.restore()

            if early_n > self.early_stop_n:
                self.logger.write("early stopped!")
                return best_loss

        return best_loss


class KLDivWithLogitsLoss(nn.KLDivLoss):
    """Kullback-Leibler divergence loss with logits as input."""

    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y_pred, y_true):
        y_pred = F.log_softmax(y_pred, dim=1)
        kldiv_loss = super().forward(y_pred, y_true)

        return kldiv_loss


def fit_model(logger, model, lr_list,
              train_dataloader, val_dataloader,
              save_dir, save_prefix, num_train_epochs,
              only_predict=False, weight_decay=0.0, early_stop_n=2,
              data_type='raw'):
    # lr_list = [1e-3, 1e-3, 1e-3, 1e-4, 1e-5]
    loss_fn = KLDivWithLogitsLoss()

    # 对除了bias和LayerNorm层的所有参数应用L2正则化
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if
                    p.requires_grad and not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    print('base lr: ', lr_list[0])
    optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters,
                                 lr=1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_list[epoch])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                               lr_lambda=lambda step: lr_list[step//len(train_dataloader)])

    val_steps_ratio = 1.0
    trainer = My_Trainer(model=model,
                         loss_fn=loss_fn,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         logger=logger,
                         save_dir=save_dir,
                         val_steps=int(val_steps_ratio * len(train_dataloader)) if not only_predict else 0,
                         log_steps=50,
                         device='cuda',
                         gradient_accum_steps=1,
                         max_grad_norm=3.0,
                         batch_to_model_inputs_fn=None,
                         early_stop_n=early_stop_n,
                         fp16=True,
                         with_ema=True,
                         save_name_prefix=save_prefix,
                         data_type=data_type)
    if only_predict:
        trainer.load_best()
        return trainer.predict(val_dataloader, return_loss=True)

    # if only_val:
    #     print('only val ..')
    #     eval_loss = trainer.val_loss(val_dataloader)
    #     print('eval: ', eval_loss)
    #     return eval_loss['eval_loss']

    score = trainer.train(train_dataloader, val_dataloader, num_train_epochs)

    # load best
    trainer.load_best()
    return trainer.predict(val_dataloader), score
