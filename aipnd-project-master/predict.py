import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

from PIL import Image
import json

import numpy as np

import argparse
import utils

parser = argparse.ArgumentParser()

parser.add_argument('--input_img', dest='input_img',
                    default="./flowers/test/1/image_06752.jpg")
parser.add_argument('--checkpoint', dest='checkpoint',
                    default='./checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', default=5)
parser.add_argument('--cat_name_json', dest='cat_name_json',
                    default='cat_to_name.json')
parser.add_argument('--gpu', dest='gpu', default='gpu')
parser.add_argument('--arch', dest='arch', default='densenet121')


args = parser.parse_args()


trainloader, testloader, validationloader, train_data = utils.load_data()
load_model = utils.load_checkpoint(args.checkpoint, args.arch, args.gpu)

top_p, top_class = utils.predict(
    args.input_img, load_model, args.top_k, args.gpu)

with open(args.cat_name_json, 'r') as json_file:
    cat_to_name = json.load(json_file)

    top_class_names = [cat_to_name[top_classes]
                       for top_classes in list(top_class)]

    print(top_p)
    print(top_class_names)
