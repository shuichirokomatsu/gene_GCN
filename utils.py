# -*- coding: utf-8 -*-

import chainer

import numpy as np
import scipy.sparse as sp

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def load_data(path="./data/", dataset="gene"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #範囲を変更
    features = sp.csr_matrix(idx_features_labels[1:, 1:-1], dtype=np.float32)
    #labels=SF2(float32)に設定、あくまでリスト
    labels = np.array(idx_features_labels[1:, -1], dtype=np.float32)
    #labels = encode_onehot(idx_features_labels[1:, -1])

    # build graph
    idx = np.array(idx_features_labels[0, 1:-1], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # ここの行列がサンプル数なのか、遺伝子の数にすべきか
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(idx.shape[0], idx.shape[0]),
                        dtype=np.float32)
    #adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        #shape=(labels.shape[0], labels.shape[0]),
                        #dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    
    #ここの数値を変える必要があるか。
    idx_train = range(24)
    idx_val = range(25, 35)
    idx_test = range(36,43)
    #初期値
    #idx_train = range(140)
    #idx_val = range(200, 500)
    #idx_test = range(500, 1500)

    features = np.array(features.todense()).astype(np.float32)
    labels = np.where(labels)[0].astype(np.float32)
    #labels = np.where(labels)[1].astype(np.int8)
    adj = sparse_mx_to_chainer_sparse_variable(adj)

    return adj, features, labels, idx_train, idx_val, idx_test

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

#Accuracyは数値であるため、誤差（プラスマイナスあり）とした
#def accuracy(output, labels):
    #preds = output.max(1)[1]
    #correct = preds - labels
    #correct = correct.sum()
    #return preds / len(labels)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_chainer_sparse_variable(sparse_mx):
    """Convert a scipy sparse matrix to a chainer sparse variable."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    data = sparse_mx.data
    row = sparse_mx.row
    col = sparse_mx.col
    shape = sparse_mx.shape
    return chainer.utils.CooMatrix(data, row, col, shape)