from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import glob
import networkx as nx
import os
import pandas as pd
from sklearn.decomposition import PCA





def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj


def sample_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def get_splits(y):
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    y_val[idx_val] = y[idx_val]
    y_test[idx_test] = y[idx_test]
    train_mask = sample_mask(idx_train, y.shape[0])
    return y_train, y_val, y_test, idx_train, idx_val, idx_test, train_mask

def get_adj_mat():

    edge_df = pd.read_csv('KnownDDI.csv')
    G = nx.from_pandas_edgelist(edge_df,source='Drug1',target='Drug2')
    drug2id = {}
    nodes = list(G.nodes)
    total_num_nodes = G.number_of_nodes()
    for i,drug in enumerate(nodes):
        drug2id[drug] = i
    nx.relabel_nodes(G,drug2id,copy=False)
   
    edges = list(G.edges)
    edges = np.array(edges)
 
    return edges,total_num_nodes

def load_feat(similarity_profile_file):
   
    init_feat = pd.read_csv(similarity_profile_file,index_col=0)
    init_feat = init_feat.to_numpy()
    # pca = PCA(0.99)
    pca = PCA(256)
    feat_mat = pca.fit_transform(init_feat)
    return feat_mat
 
def get_train_test_set(edges_all,num_nodes,ratio=0.2):
    
    
    np.random.shuffle(edges_all)
    test_size=int(edges_all.shape[0]*ratio)
    test_edges_true=edges_all[0:test_size]
    
    train_edges_true=edges_all[test_size:]
   
    df = pd.read_csv('./negative_samples.csv',index_col=0)
    samples = df.to_numpy()
    np.random.shuffle(samples)
    test_false_size=int(samples.shape[0]*ratio)
    test_edges_false = samples[:test_false_size,:-1].astype(np.int32)
    train_edges_false = samples[test_false_size:,:-1].astype(np.int32)
    data = np.ones(train_edges_true.shape[0])

    # Re-build adj matrix
    adj_train = sp.csr_matrix((data, (train_edges_true[:, 0], train_edges_true[:, 1])), shape=(num_nodes,num_nodes),dtype=np.float32)
    adj_train = adj_train + adj_train.T
    
    # NOTE: these edge lists only contain single direction of edge!
    return adj_train, train_edges_true, train_edges_false, test_edges_true, test_edges_false


def get_full_dataset():
    
    
    
    edge_df = pd.read_csv('KnownDDI.csv')
    G = nx.from_pandas_edgelist(edge_df,source='Drug1',target='Drug2')
    drug2id = {}
    nodes = list(G.nodes)
    for i,drug in enumerate(nodes):
        drug2id[drug] = i
    nx.relabel_nodes(G,drug2id,copy=False)
    edges_all = list(G.edges)
    edges_all = np.array(edges_all)
    np.random.shuffle(edges_all)
    y_true = np.ones(edges_all.shape[0],dtype=np.int8)

   
    df = pd.read_csv('./negative_samples.csv',index_col=0)
    samples = df.to_numpy()
    np.random.shuffle(samples)

    false_edges = samples[:,:-1]
    y_false = np.zeros(false_edges.shape[0],dtype=np.int8)
   
    return np.vstack([edges_all,false_edges]),np.concatenate([y_true,y_false])
