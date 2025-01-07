import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm


class CNN_block(nn.Module):
  def __init__(self, input_channels, output_channels, **kwargs):
    super(CNN_block, self).__init__()
    self.conv = nn.Conv2d(input_channels, output_channels, bias=False, **kwargs)
    self.batchnorm = nn.BatchNorm2d(output_channels)
    self.leakyrelu = nn.LeakyReLU(0.1)

  def forward(self, x):
    return self.leakyrelu(self.batchnorm(self.conv(x)))


class CharaterRecognizer(nn.Module):
  def __init__(self, nums_classifier, cnn_config):
    super(CharaterRecognizer, self).__init__()
    self.nums_classifier = nums_classifier
    self.cnn, self.nums_M = self._build_cnn(cnn_config)
    self.classifer = self._build_classifer()

  def _build_cnn(self, cnn_config):
    input_channels = 1 # color channel of the image
    count_M = 0
    layers = []
    for layer_config in cnn_config:
      if isinstance(layer_config, list):
        for time in range(layer_config[-1]):
          for x in layer_config[:-1]:
            if (isinstance(x, tuple)):
              layers += [
                  CNN_block(input_channels, x[0], kernel_size=x[1], stride=x[2], padding=x[3])
              ]
              input_channels = x[0]
      if isinstance(layer_config, str):
        if (layer_config == 'M'):
          count_M += 1
          layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        elif (layer_config == 'D'):
          layers += [nn.Dropout(0.25)]

    return nn.Sequential(*layers), count_M

  def _build_classifer(self):
    layers = [
        nn.Flatten(),
        nn.Linear(3136, 256),
        nn.Dropout(0.25),
        nn.LeakyReLU(0.1),
        nn.Linear(256, self.nums_classifier),
        nn.Softmax(dim=1)

    ]
    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.cnn(x)
    x = self.classifer(x)
    return x

