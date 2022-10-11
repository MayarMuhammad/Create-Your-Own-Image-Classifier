import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image

import numpy as np

import argparse
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', dest='data_dir', default="./flowers/")
parser.add_argument('--save_dir', dest='save_dir', default='./checkpoint.pth')
parser.add_argument('--dropout', dest='dropout', default=0.5)
parser.add_argument('--learning_rate', dest='learning_rate', default=0.001)
parser.add_argument('--hidden_layer_1', dest='hidden_layer_1', default=512)
parser.add_argument('--hidden_layer_2', dest='hidden_layer_2', default=256)
parser.add_argument('--epochs', dest='epochs', default=5)
parser.add_argument('--gpu', dest='gpu', default='gpu')
parser.add_argument('--arch', dest='arch', default='densenet121')

args = parser.parse_args()

trainloader, testloader, validationloader, train_data = utils.load_data(args.data_dir)
model, optimizer, criterion = utils.network_setup(args.arch, args.gpu,args.learning_rate,args.hidden_layer_1,args.hidden_layer_2,args.dropout)
utils.train_network(model, optimizer, criterion, args.epochs, 20, args.gpu)
utils.save_checkpoint(args.arch, args.gpu)

print('Model Trained')
