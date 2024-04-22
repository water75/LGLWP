import sys
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import pdb
from mlp_dropout import MLPClassifier, MLPRegression
from sklearn import metrics
from util import cmd_args, load_data
from sklearn.metrics import average_precision_score

def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(list(range(total_iters)), unit='batch')
    all_targets = []
    all_scores = []

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        targets = [g_list[idx].label for idx in selected_idx]
        all_targets += targets
        if classifier.regression:
            pred, mae, loss = classifier(batch_graph)
            all_scores.append(pred.cpu().detach())  # for binary classification
        else:
            logits, loss, acc = classifier(batch_graph)
            all_scores.append(logits[:, 1].cpu().detach())  # for binary classification

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()
        if classifier.regression:
            pbar.set_description('MSE_loss: %0.5f MAE_loss: %0.5f' % (loss, mae) )
            total_loss.append( np.array([loss, mae]) * len(selected_idx))
        else:
            pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )
            total_loss.append( np.array([loss, acc]) * len(selected_idx))


        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    
    if not classifier.regression:
        all_targets = np.array(all_targets)
        avg_precision = average_precision_score(all_targets, all_scores)
        fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        avg_loss = np.concatenate((avg_loss, [auc, avg_precision]))
    
    return avg_loss


def loop_dataset_gem(classifier, loader, optimizer=None):
    all_targets = []
    all_pre = []
    all_y = []

    pbar = tqdm(loader, unit='batch')
    n_samples = 0
    for batch in pbar:
        all_targets.extend(batch.y)
        pre, loss, mae, y = classifier(batch)
        all_y.extend(y.data.cpu().detach().numpy())
        all_pre.extend(pre.data.cpu().detach().numpy())
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.data.cpu().detach().numpy()

        pbar.set_description('mse: %0.5f ' % (loss) )
        # total_loss.append( np.array([loss, acc]) * len(batch.y))
        
        n_samples += len(batch.y)

    m = np.mean((np.array(all_y) - np.array(all_pre))** 2)  # 计算均方误差
    rmse = np.sqrt(m)  # 计算均方根误差
    # if optimizer is None:
    #     print(all_targets)


    # total_loss = np.array(total_loss)
    # avg_loss = np.sum(total_loss, 0) / n_samples
    # all_scores = torch.cat(all_scores).cpu().numpy()
    #
    # # np.savetxt('test_scores.txt', all_scores)  # output test predictions
    #
    # all_targets = np.array(all_targets)
    # # 精确度
    # avg_precision = average_precision_score(all_targets, all_scores)
    # fpr, tpr, _ = metrics.roc_curve(all_targets, all_scores, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    # avg_loss = np.concatenate((avg_loss, [auc, avg_precision]))
    #
    return rmse


