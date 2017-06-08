import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.cuda
from pprint import pprint, pformat
import pickle
import argparse
import os
import math
import matplotlib.pyplot as plt

from pytorch_model import ProdLDA
from pytorch_visualize import *

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--en1-units',        type=int,   default=100)
parser.add_argument('-s', '--en2-units',        type=int,   default=100)
parser.add_argument('-t', '--num-topic',        type=int,   default=50)
parser.add_argument('-b', '--batch-size',       type=int,   default=200)
parser.add_argument('-o', '--optimizer',        type=str,   default='Adam')
parser.add_argument('-r', '--learning-rate',    type=float, default=0.002)
parser.add_argument('-m', '--momentum',         type=float, default=0.99)
parser.add_argument('-e', '--num-epoch',        type=int,   default=80)
parser.add_argument('-q', '--init-mult',        type=float, default=1.0)    # multiplier in initialization of decoder weight
parser.add_argument('-v', '--variance',         type=float, default=0.995)  # default variance in prior normal
parser.add_argument('--start',                  action='store_true')        # start training at invocation
parser.add_argument('--nogpu',                  action='store_true')        # do not use GPU acceleration

args = parser.parse_args()


# default to use GPU, but have to check if GPU exists
if not args.nogpu:
    if torch.cuda.device_count() == 0:
        args.nogpu = True

def to_onehot(data, min_length):
    return np.bincount(data, minlength=min_length)

def make_data():
    global data_tr, data_te, tensor_tr, tensor_te, vocab, vocab_size
    dataset_tr = 'data/20news_clean/train.txt.npy'
    data_tr = np.load(dataset_tr)
    dataset_te = 'data/20news_clean/test.txt.npy'
    data_te = np.load(dataset_te)
    vocab = 'data/20news_clean/vocab.pkl'
    vocab = pickle.load(open(vocab,'r'))
    vocab_size=len(vocab)
    #--------------convert to one-hot representation------------------
    print 'Converting data to one-hot representation'
    data_tr = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_tr if np.sum(doc)!=0])
    data_te = np.array([to_onehot(doc.astype('int'),vocab_size) for doc in data_te if np.sum(doc)!=0])
    #--------------print the data dimentions--------------------------
    print 'Data Loaded'
    print 'Dim Training Data',data_tr.shape
    print 'Dim Test Data',data_te.shape
    #--------------make tensor datasets-------------------------------
    tensor_tr = torch.from_numpy(data_tr).float()
    tensor_te = torch.from_numpy(data_te).float()
    if not args.nogpu:
        tensor_tr = tensor_tr.cuda()
        tensor_te = tensor_te.cuda()

def make_model():
    global model
    net_arch = args # en1_units, en2_units, num_topic, num_input
    net_arch.num_input = data_tr.shape[1]
    model = ProdLDA(net_arch)
    if not args.nogpu:
        model = model.cuda()

def make_optimizer():
    global optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, betas=(args.momentum, 0.999))
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum)
    else:
        assert False, 'Unknown optimizer {}'.format(args.optimizer)

def train():
    for epoch in xrange(args.num_epoch):
        all_indices = torch.randperm(tensor_tr.size(0)).split(args.batch_size)
        loss_epoch = 0.0
        model.train()                   # switch to training mode
        for batch_indices in all_indices:
            if not args.nogpu: batch_indices = batch_indices.cuda()
            input = Variable(tensor_tr[batch_indices])
            recon, loss = model(input, compute_loss=True)
            # optimize
            optimizer.zero_grad()       # clear previous gradients
            loss.backward()             # backprop
            optimizer.step()            # update parameters
            # report
            loss_epoch += loss.data[0]    # add loss to loss_epoch
        if epoch % 5 == 0:
            print('Epoch {}, loss={}'.format(epoch, loss_epoch / len(all_indices)))

associations = {
    'jesus': ['prophet', 'jesus', 'matthew', 'christ', 'worship', 'church'],
    'comp ': ['floppy', 'windows', 'microsoft', 'monitor', 'workstation', 'macintosh', 
              'printer', 'programmer', 'colormap', 'scsi', 'jpeg', 'compression'],
    'car  ': ['wheel', 'tire'],
    'polit': ['amendment', 'libert', 'regulation', 'president'],
    'crime': ['violent', 'homicide', 'rape'],
    'midea': ['lebanese', 'israel', 'lebanon', 'palest'],
    'sport': ['coach', 'hitter', 'pitch'],
    'gears': ['helmet', 'bike'],
    'nasa ': ['orbit', 'spacecraft'],
}
def identify_topic_in_line(line):
    topics = []
    for topic, keywords in associations.iteritems():
        for word in keywords:
            if word in line:
                topics.append(topic)
                break
    return topics

def print_top_words(beta, feature_names, n_top_words=10):
    print '---------------Printing the Topics------------------'
    for i in range(len(beta)):
        line = " ".join([feature_names[j] 
                            for j in beta[i].argsort()[:-n_top_words - 1:-1]])
        topics = identify_topic_in_line(line)
        print('|'.join(topics))
        print('     {}'.format(line))
    print '---------------End of Topics------------------'

def print_perp(model):
    cost=[]
    model.eval()                        # switch to testing mode
    input = Variable(tensor_te)
    recon, loss = model(input, compute_loss=True, avg_loss=False)
    loss = loss.data
    counts = tensor_te.sum(1)
    avg = (loss / counts).mean()
    print('The approximated perplexity is: ', math.exp(avg))

def visualize():
    global recon
    input = Variable(tensor_te[:10])
    register_vis_hooks(model)
    recon = model(input, compute_loss=False)
    remove_vis_hooks()
    save_visualization('pytorch_model', 'png')

if __name__=='__main__' and args.start:
    make_data()
    make_model()
    make_optimizer()
    train()
    emb = model.decoder.weight.data.cpu().numpy().T
    print_top_words(emb, zip(*sorted(vocab.items(), key=lambda x:x[1]))[0])
    print_perp(model)
    visualize()

