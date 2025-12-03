import time
import torch
from scipy import stats
import numpy as np
from base_tools import *
from baseline import Baseline
from DRCDown import DRCDown


class BaseIQASolver(object):
    def __init__(self, config):

        self.test_patch_num = config.test_patch_num
        self.dataset = config.dataset

        self.model = Baseline(pretrain=config.pretrain, output_f=True).cuda()

        self.l1_loss = torch.nn.L1Loss().cuda()

        backbone_params = list(map(id, self.model.backbone.parameters()))
        self.hypernet_params = filter(lambda p: id(p) not in backbone_params, self.model.parameters())
        self.lr = config.lr
        self.lrratio = config.lr_ratio
        self.weight_decay = config.weight_decay
        paras = [{'params': self.hypernet_params, 'lr': self.lr * self.lrratio},
                 {'params': self.model.backbone.parameters(), 'lr': self.lr}
                 ]
        self.solver = torch.optim.Adam(paras, weight_decay=self.weight_decay)

    def downsample(self, train_data):
        self.model.eval()
        pred_scores = []
        gt_scores = []
        f_list = []
        for _, img, label in train_data:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                pred, f = self.model(img)
            f_list.append(f.cpu().detach().numpy())

            pred_scores += pred.cpu().tolist()
            gt_scores += label.cpu().tolist()

        filtered_idx = DRCDown(gt_scores, f_list)
        return filtered_idx

    def train(self, train_data):
        self.model.train()

        epoch_loss = []
        pred_scores = []
        gt_scores = []

        for _, img, label in train_data:
            img = img.cuda()
            label = label.cuda()

            self.solver.zero_grad()

            pred, _ = self.model(img)
            loss = self.l1_loss(pred.squeeze(), label.float().detach())  # * gamma_lr

            pred_scores += pred.cpu().tolist()
            gt_scores += label.cpu().tolist()
            epoch_loss.append(loss.item())

            loss.backward()
            self.solver.step()

        train_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        return epoch_loss, train_srcc

    def test(self, data):
        self.model.eval()
        pred_scores = []
        gt_scores = []
        f_list = []
        name_list = []

        for imgname, img, label in data:
            img = img.cuda()
            label = label.cuda()

            with torch.no_grad():
                pred, f = self.model(img)

            f_list.append(f.cpu().detach().numpy())

            name_list += list(imgname)
            pred_scores += pred.cpu().tolist()
            gt_scores += label.cpu().tolist()

        pred_scores = np.mean(np.reshape(np.array(pred_scores), (-1, self.test_patch_num)), axis=1)
        gt_scores = np.mean(np.reshape(np.array(gt_scores), (-1, self.test_patch_num)), axis=1)
        test_srcc, _ = stats.spearmanr(pred_scores, gt_scores)
        test_plcc, _ = stats.pearsonr(pred_scores, gt_scores)

        return test_srcc, test_plcc, (name_list[0::self.test_patch_num], f_list, pred_scores, gt_scores)
