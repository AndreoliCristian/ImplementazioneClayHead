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


class DecoupledYOLOHead(nn.Module):
    def __init__(self, num_classes, num_anchors=3, hidden_channels=256):
        super(DecoupledYOLOHead, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        # Branch for bounding box regression + objectness
        # Output channels: num_anchors * 5 -> (x, y, w, h, obj_score)
        self.bbox_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.num_anchors * 5, kernel_size=1)
        )

        # Branch for classification
        # Output channels: num_anchors * num_classes
        self.cls_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.num_anchors * self.num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x: [B, hidden_channels, H, W]
        bbox_out = self.bbox_conv(x)  # [B, A*5, H, W]
        cls_out = self.cls_conv(x)  # [B, A*C, H, W]

        # Puoi restituire due output separati e poi combinarli in post-processing
        return bbox_out, cls_out


class MLPHead(nn.Module):
    def __init__(self, input_dim, num_classes, num_boxes=100):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        # output_dim = (x,y,w,h,obj) + num_classes
        self.output_dim = (5 + self.num_classes) * self.num_boxes

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.output_dim)
        )

    def forward(self, x):
        # x: [B, L*D]
        out = self.mlp(x)  # [B, N*(5+num_classes)]
        out = out.view(x.size(0), self.num_boxes, 5 + self.num_classes)
        # out: [B, N, 5+num_classes]
        return out


