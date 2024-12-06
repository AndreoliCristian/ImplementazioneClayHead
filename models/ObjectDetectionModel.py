# models/__init__.py

import torch
import pytorch_lightning as pl
from models.encoder import PretrainedEncoder
from models.detector import DetectionHead


class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, encoder_weights_path, num_classes, learning_rate=1e-3):
        super(ObjectDetectionModel, self).__init__()
        self.encoder = PretrainedEncoder(encoder_weights_path)
        self.detector = DetectionHead(num_classes, input_dim=encoder_output_dim)
        self.learning_rate = learning_rate

        # Loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.encoder(x)
        # Reshape features if necessary
        features = features.permute(0, 2, 1).view(-1, encoder_output_dim, h, w)
        output = self.detector(features)
        return output

    def training_step(self, batch, batch_idx):
        images = batch['pixels']
        targets = batch['targets']  # Adjust this based on your target format

        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['pixels']
        targets = batch['targets']
        outputs = self(images)
        loss = self.loss_fn(outputs, targets)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.detector.parameters(), lr=self.learning_rate)
        return optimizer
