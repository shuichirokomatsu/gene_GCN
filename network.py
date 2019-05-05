# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
from chainer import reporter

import numpy as np

from graph_convolution import GraphConvolution

# Network definition
class GraphConvolutionalNetwork(chainer.Chain):
    def __init__(self, n_input, n_mid, n_out, adj, idx_train, idx_val,
                 test_mode=False):
        super(GraphConvolutionalNetwork, self).__init__()
        with self.init_scope():
            self.gconv1 = GraphConvolution(n_input, n_mid)
            self.gconv2 = GraphConvolution(n_mid, n_out)
            
            self.n_input = n_input
        
            self.adj = adj
            self.idx_train = idx_train
            self.idx_val = idx_val
            
            self.test_mode = test_mode
            
    def __call__(self, inputs):
        x = inputs[:, :self.n_input]
        #回帰問題のためfloat32に変更
        labels = inputs[:, self.n_input:][:,0].astype(np.int32)
        
        h = F.relu(self.gconv1(x, self.adj))
        h = F.dropout(h, 0.5)    
        out = self.gconv2(h, self.adj)
        
        if chainer.config.train == True:
            loss = F.softmax_cross_entropy(out[self.idx_train], labels[self.idx_train])
            accuracy = F.accuracy(out[self.idx_train], labels[self.idx_train]) 
            #回帰のLossのため、mean_squareに変更
            #loss = F.mean_squared_error(out[self.idx_train], labels[self.idx_train])
            #accuracy = F.mean_absolute_error(out[self.idx_train], labels[self.idx_train]) 
        else:
            #loss = F.mean_squared_error(out[self.idx_val], labels[self.idx_val])
            #accuracy = F.mean_absolute_error(out[self.idx_val], labels[self.idx_val])
            loss = F.softmax_cross_entropy(out[self.idx_val], labels[self.idx_val])
            accuracy = F.accuracy(out[self.idx_val], labels[self.idx_val])
            
        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': accuracy}, self)
        
        if self.test_mode == True:
            return loss, accuracy
        else:
            return loss