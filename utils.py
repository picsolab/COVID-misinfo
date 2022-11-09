# coding: utf-8

import numpy as np
from sklearn import metrics
import torch.nn.functional as F

def perf(output, target):
    y_true = target.cpu()
    y_prob = F.softmax(output, dim=-1)[:,1].cpu()
    y_pred = F.softmax(output, dim=-1).argmax(dim=-1).cpu()
    acc = metrics.accuracy_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
    f1 = 2. * pre * rec / (pre + rec)
    return acc,auc,pre,rec,f1