import torch
import numpy as np
import sys, copy, math, time, pdb
import pickle as pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
# sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from data_sea import dataset
from main import *
from util_functions import *
from torch_geometric.data import DataLoader
from model import Net
import pandas as pd
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='Link Prediction')
# general settings
parser.add_argument('--data-name', default='BUP', help='network name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=10000, 
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.5,
                    help='ratio of test links')
# model settings
parser.add_argument('--hop', default=2, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=100, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')




args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)

# total = []
# for number in range(7):
allresults = []
allcorrs = []
# + str(number)
for m in range(10):
    Matrix, Adj, Node, train_pos = dataset.Read_graph('data_sea/neural/neural2' + str(m), 296)
    samdata = pd.read_csv('data_sea/neural/neuralsam' + str(m) + '.csv')
    adj_mat = Adj.clone()
    for e in samdata.iloc[:, :-1].values:
        adj_mat[e[0], e[1]] = 1
        adj_mat[e[1], e[0]] = 1
    samdata['2'] = np.exp(-1 / samdata['2'])
    edge = samdata.iloc[:, :-1].values
    min_node, max_node = 0, edge.max()
    Matrix_copy = Matrix.clone()
    for e in samdata.iloc[:, :].values:
        if min_node == 0:
            Matrix[int(e[0]), int(e[1])] = e[2]
            Matrix[int(e[1]), int(e[0])] = e[2]
        else:
            Matrix[int(e[0]) - 1, int(e[1]) - 1] = e[2]
            Matrix[int(e[1]) - 1, int(e[0]) - 1] = e[2]

    samdata = samdata.values
    test_pos = (samdata[:, 0].astype(np.int32), samdata[:, 1].astype(np.int32))
    # 分界线
    '''Train and apply classifier'''

    A = csc_matrix(adj_mat)
    # 创建一个无向图
    G = nx.Graph()

    # 添加节点
    num_nodes = Matrix.shape[0]
    G.add_nodes_from(range(num_nodes))

    # 添加边和权重
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = Matrix[i, j]
            if weight != 0:
                G.add_edge(i, j, weight=weight)

    train_graphs, test_graphs, max_n_label = links2subgraphs(G, A, train_pos, test_pos ,Matrix)
    print(('train: %d, test: %d' % (len(train_graphs), len(test_graphs))))

    train_lines = to_linegraphs(train_graphs, max_n_label)
    test_lines = to_linegraphs(test_graphs, max_n_label)

    # Model configurations
    cmd_args.latent_dim = [32, 32, 32, 1]
    cmd_args.hidden = 128
    cmd_args.out_dim = 0
    cmd_args.dropout = True
    cmd_args.num_class = 2
    cmd_args.mode = 'gpu'
    cmd_args.num_epochs = 15
    cmd_args.learning_rate = 5e-3
    cmd_args.batch_size = 50
    cmd_args.printAUC = True
    cmd_args.feat_dim = max_n_label * 2
    cmd_args.attr_dim = 0

    train_loader = DataLoader(train_lines, batch_size=cmd_args.batch_size, shuffle=True)
    test_loader = DataLoader(test_lines, batch_size=cmd_args.batch_size, shuffle=False)

    classifier = Net(cmd_args.feat_dim, cmd_args.hidden, cmd_args.latent_dim, with_dropout=cmd_args.dropout)
    if cmd_args.mode == 'gpu':
        classifier = classifier.to("cuda")

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    best_auc = 0
    best_auc_acc = 0
    best_acc = 0
    best_acc_auc = 0
    all_loss = []
    all_test_loss = []
    for epoch in range(cmd_args.num_epochs):
        classifier.train()
        avg_loss = loop_dataset_gem(classifier, train_loader, optimizer=optimizer)
        print(('\033[92m第%d个 average training of epoch %d: loss %.5f \033[0m' % (m+1, epoch, avg_loss)))
        all_loss.append(avg_loss)
        classifier.eval()
        test_loss = loop_dataset_gem(classifier, test_loader, None)
        all_test_loss.append(test_loss)
