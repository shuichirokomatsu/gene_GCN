# -*- coding: utf-8 -*-

import argparse

import chainer
from chainer import training
from chainer.training import extensions
from chainer import iterators, optimizers, serializers
import numpy as np
from chainer import functions as F

import network

from utils import load_data
#from regressor import Regressor

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', type=int, default=-1)
parser.add_argument('--model', '-m', type=str, default=None)
parser.add_argument('--epoch', '-e', type=int, default=200)
parser.add_argument('--lr', '-l', type=float, default=0.01)
parser.add_argument('--noplot', dest='plot', action='store_false',
                    help='Disable PlotReport extension')
args = parser.parse_args()

print("Loading datas")
# Get data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

#ラベルをScalerにするかいなか、しない
#scaler = None

# Accuracyの代わり
class MeanAbsError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) mean absolute error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if self.scaler is not None:
            x0 = self.scaler.inverse_transform(x0)
            x1 = self.scaler.inverse_transform(x1)
        return F.mean_absolute_error(x0, x1)

class RootMeanSqrError(object):
    def __init__(self, scaler=None):
        """Initializes the (scaled) root mean square error metric object.
        Args:
            scaler: Standard label scaler.
        """
        self.scaler = scaler

    def __call__(self, x0, x1):
        if self.scaler is not None:
            x0 = self.scaler.inverse_transform(x0)
            x1 = self.scaler.inverse_transform(x1)
        return F.sqrt(F.mean_squared_error(x0, x1))

# Normalize X
features /= features.sum(1).reshape(-1, 1)
labels = np.expand_dims(labels, 1)

# Inputs
inputs = np.concatenate((features, labels), axis=1)

#batch_sizeを変えた。adj.shape[0]
train_iter = iterators.SerialIterator(inputs, batch_size=30, shuffle=False)
test_iter = iterators.SerialIterator(inputs, batch_size=30, repeat=False, shuffle=False)

# Set up a neural network to train.
print("Building model")
#遺伝子480、Outputは数字のみ
model = network.GraphConvolutionalNetwork(480, 16, 1, adj, idx_train, idx_val)

if args.gpu >= 0:
    # Make a specified GPU current
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    model.to_gpu()  # Copy the model to the GPU

#Regressorの導入
#metrics_fun = {'mae': MeanAbsError(scaler=scaler), 'rmse': RootMeanSqrError(scaler=scaler)}
#regressor = Regressor(model, lossfun=F.mean_squared_error, metrics_fun=metrics_fun, device=args.gpu)

optimizer = optimizers.Adam(alpha=args.lr)
optimizer.setup(model)
#optimizerをRegressorに変更
#optimizer.setup(regressor)
#optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

if args.model != None:
    print( "loading model from " + args.model)
    serializers.load_npz(args.model, model)

updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
trainer = training.Trainer(updater, (args.epoch, 'epoch'), out='results')

trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
#modelをRegressorに変更
#trainer.extend(extensions.Evaluator(test_iter, regressor, device=args.gpu))
trainer.extend(extensions.LogReport())

# Save two plot images to the result dir
if args.plot and extensions.PlotReport.available():
    trainer.extend(
        extensions.PlotReport(['main/loss', 'validation/main/loss'],
                              'epoch', file_name='loss.png'), trigger=(10, 'epoch'))
    trainer.extend(
        extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'],
            'epoch', file_name='accuracy.png'), trigger=(10, 'epoch'))

trainer.extend(extensions.PrintReport(
    ['epoch', 'main/loss', 'validation/main/loss',
     'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

trainer.extend(extensions.ProgressBar())

# Train
trainer.run()

# Save results
print("Optimization Finished!")
modelname = "./results/model"
print( "Saving model to " + modelname)
serializers.save_npz(modelname, model)

# Test
model = network.GraphConvolutionalNetwork(480, 16, 1, adj, None, idx_test, True)
serializers.load_npz("./results/model", model)
with chainer.using_config('train', False):
    loss_test, acc_test = model(inputs)
print("Test set results:\n",
      "loss =", loss_test.data,
      "\n accuracy =", acc_test.data)
