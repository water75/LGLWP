import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import pickle as cp
#import _pickle as cp  # python3 compatability
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
import scipy.sparse as sp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
from data_processor import get_subgraph
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append('%s/../../pytorch_DGCNN' % cur_dir)
import multiprocessing as mp
import torch

from torch_geometric.transforms import LineGraph
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import from_networkx

class GNNGraph(object):
    def __init__(self, g, label, node_tags=None, node_features=None):
        '''
            g: a networkx graph
            label: an integer graph label
            node_tags: a list of integer node tags
            node_features: a numpy array of continuous node features
        '''
        self.num_nodes = len(g.nodes)
        self.node_tags = node_tags
        self.label = label
        self.node_features = node_features  # numpy array (node_num * feature_dim)
        self.degs = list(dict(g.degree).values())
        if len(g.edges()) != 0:
            x, y = list(zip(*g.edges()))
            self.num_edges = len(x)
            self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
            self.edge_pairs[:, 0] = x
            self.edge_pairs[:, 1] = y
            self.edge_pairs = self.edge_pairs.flatten()
        else:
            self.num_edges = 0
            self.edge_pairs = np.array([])
        
        # see if there are edge features
        self.edge_features = None
        if nx.get_edge_attributes(g, 'features'):  
            # make sure edges have an attribute 'features' (1 * feature_dim numpy array)
            edge_features = nx.get_edge_attributes(g, 'features')
            assert(type(list(edge_features.values())[0]) == np.ndarray) 
            # need to rearrange edge_features using the e2n edge order
            edge_features = {(min(x, y), max(x, y)): z for (x, y), z in list(edge_features.items())}
            keys = sorted(edge_features)
            self.edge_features = []
            for edge in keys:
                self.edge_features.append(edge_features[edge])
                self.edge_features.append(edge_features[edge])  # add reversed edges
            self.edge_features = np.concatenate(self.edge_features, 0)

def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None, max_train_num=None):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None or test_pos is None:
        perm = random.sample(list(range(len(row))), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative links for train/test
    train_num, test_num = len(train_pos[0]), len(test_pos[0])
    print(train_num, test_num)
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    while len(neg[0]) < train_num + test_num:
        i, j = random.randint(0, n-1), random.randint(0, n-1)
        if i < j and net[i, j] == 0:
            neg[0].append(i)
            neg[1].append(j)
        else:
            continue
    train_neg  = (neg[0][:train_num], neg[1][:train_num])
    test_neg = (neg[0][train_num:], neg[1][train_num:])
    return train_pos, train_neg, test_pos, test_neg

    
def links2subgraphs(G, A, train_pos, test_pos, Matrix):
    # extract enclosing subgraphs
    max_n_label = {'value': 0}
    def helper(A, links):
        g_list = []
        for i, j in tqdm(zip(links[0], links[1]),desc='子图提取'):
            g, n_features = subgraph_extraction_labeling(G, (i, j), A)
            max_n_label['value'] = max(len(g.nodes), max_n_label['value'])
            g_list.append(GNNGraph(g, Matrix[i][j], None, n_features))
        return g_list
    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(A, train_pos)
    test_graphs = helper(A, test_pos)
    print('Enclosing subgraph extraction over...')
    return train_graphs, test_graphs, max_n_label['value']

def subgraph_extraction_labeling(G, ind, A):
    g, subgraph_matrix = get_subgraph(G, A, ind[0], ind[1], False, 10)

    return g, subgraph_matrix

def to_linegraphs(batch_graphs, max_n_label):
    graphs = []
    pbar = tqdm(batch_graphs, unit='iteration')
    for graph in pbar:
        # edges = graph.edge_pairs
        # edge_feas = edge_fea(graph, max_n_label)/2
        # edges, feas = to_undirect(edges, edge_feas)
        # edges = torch.tensor(edges)
        # feas = torch.tensor(feas)
        # feas = feas.to(dtype=torch.float32)
        # data = Data(edge_index=edges, edge_attr=feas)
        # data.num_nodes = graph.num_nodes
        # data = LineGraph()(data)
        # data['y'] = torch.tensor([graph.label])
        # data.num_nodes = graph.num_edges
        # graphs.append(data)
        edges = graph.edge_pairs
        edge_feas = edge_fea(graph, max_n_label) / 2
        edges, feas = to_undirect(edges, edge_feas)
        edges = torch.tensor(edges)
        data = Data(edge_index=edges, edge_attr=feas)
        data.num_nodes = graph.num_nodes
        data = LineGraph()(data)
        data['y'] = torch.tensor([graph.label])
        data.num_nodes = graph.num_edges
        graphs.append(data)
    return graphs

def edge_fea(graph, max_n_label):
    # node_tag = torch.zeros(graph.num_nodes, max_n_label+1)
    # tags = graph.node_tags
    # tags = torch.LongTensor(tags).view(-1,1)
    # node_tag.scatter_(1, tags, 1)
    # node_tag = torch.cat([node_tag, graph.node_features], 1)

    node_tag = graph.node_features.copy()
    # 检查第一维是否小于 max_n_label
    if node_tag.shape[1] < max_n_label:
        # 计算需要补零的数量
        num_zeros_to_add = max_n_label - node_tag.shape[1]

        # 创建一个包含零的 NumPy 数组
        zeros_to_add = np.zeros((node_tag.shape[0], num_zeros_to_add))

        # 水平堆叠到 node_tag 后面
        node_tag = np.hstack((node_tag, zeros_to_add))


    return node_tag

    
def to_undirect(edges, edge_fea):
    edges = np.reshape(edges, (-1, 2))
    sr = np.array([edges[:, 0], edges[:, 1]], dtype=np.int64)
    fea_s = edge_fea[sr[0, :], :]
    # fea_s = fea_s.repeat(2, 1)
    fea_s = np.tile(fea_s, (2, 1))

    fea_r = edge_fea[sr[1, :], :]
    # fea_r = fea_r.repeat(2, 1)
    fea_r = np.tile(fea_r, (2, 1))
    fea_s = torch.tensor(fea_s, dtype=torch.float32)
    fea_r = torch.tensor(fea_r, dtype=torch.float32)
    fea_body = torch.cat([fea_s, fea_r], 1)
    rs = np.array([edges[:, 1], edges[:, 0]], dtype=np.int64)
    return np.concatenate([sr, rs], axis=1), fea_body



