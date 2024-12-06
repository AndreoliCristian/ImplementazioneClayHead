# models/detector.py

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, num_classes, input_dim):
        super(DetectionHead, self).__init__()
        # Define detection layers
        self.conv = nn.Conv2d(input_dim, 256, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pred = nn.Conv2d(256, num_classes + 4, kernel_size=1)  # +4 for bbox coords

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        output = self.pred(x)
        return output
