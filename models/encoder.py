# models/encoder.py

import torch
import torch.nn as nn


class PretrainedEncoder(nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super(PretrainedEncoder, self).__init__()
        # If no pretrained weights are provided, initialize the encoder
        
        # Load the pretrained encoder
        self.encoder = torch.load(pretrained_weights_path)
        self.freeze_encoder()

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Forward pass through the encoder
        # Assuming encoder returns a tuple
        encoded_unmasked_patches, _, _, _ = self.encoder(x)
        # Exclude the class token at the beginning
        encoded_unmasked_patches = encoded_unmasked_patches[:, 1:, :]
        return encoded_unmasked_patches
