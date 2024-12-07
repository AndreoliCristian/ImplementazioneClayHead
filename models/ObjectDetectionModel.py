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


class ObjectDetectionModel(pl.LightningModule):
    def __init__(self, encoder, num_classes, num_boxes=100, lr=1e-3, conf_threshold=0.5, iou_threshold=0.5):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.lr = lr
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Si assume che encoder.embedding_dim e encoder.num_patches siano attributi noti
        input_dim = self.encoder.embedding_dim * self.encoder.num_patches
        self.head = MLPHead(input_dim, num_classes, num_boxes)

        self.matcher = HungarianMatcher(cost_box=1.0, cost_class=1.0)
        self.criterion = SetCriterion(num_classes, self.matcher, lambda_box=1.0, lambda_obj=1.0, lambda_class=1.0)

    def forward(self, x):
        """
        x: immagini [B, C, H, W]
        encoder: [B, L, D]
        """
        feats = self.encoder(x)  # [B, L, D]
        B, L, D = feats.shape
        flat_feats = feats.view(B, L*D)
        preds = self.head(flat_feats)
        return preds

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self(images) # [B, N, 5+num_classes]

        # Decomponi pred
        pred_boxes = preds[..., :4]
        pred_obj = preds[..., 4]
        pred_class = preds[..., 5:]
        outputs = {
            "pred_boxes": pred_boxes,
            "pred_obj": pred_obj,
            "pred_class": pred_class
        }

        loss_dict = self.criterion(outputs, targets)
        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict["loss"]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images = batch
        preds = self(images) # [B, N, 5+num_classes]

        results = []
        for i in range(preds.size(0)):
            single_pred = preds[i]  # [N, 5+num_classes]
            boxes = single_pred[..., :4]
            objectness = torch.sigmoid(single_pred[..., 4])
            class_logits = single_pred[..., 5:]
            class_probs = torch.softmax(class_logits, dim=-1)

            # Trova la classe con max score
            cls_scores, cls_labels = class_probs.max(dim=1)
            scores = objectness * cls_scores

            # Applica conf threshold
            mask = scores > self.conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            cls_labels = cls_labels[mask]

            if boxes.numel() > 0:
                x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x1 = x_c - w/2.0
                y1 = y_c - h/2.0
                x2 = x_c + w/2.0
                y2 = y_c + h/2.0
                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # NMS
                keep = nms(boxes_xyxy, scores, self.iou_threshold)
                boxes_xyxy = boxes_xyxy[keep]
                scores = scores[keep]
                cls_labels = cls_labels[keep]
            else:
                boxes_xyxy = torch.empty((0,4), device=self.device)
                scores = torch.empty(0, device=self.device)
                cls_labels = torch.empty(0, device=self.device)

            results.append({
                "boxes": boxes_xyxy.cpu(),
                "scores": scores.cpu(),
                "labels": cls_labels.cpu()
            })
        return results


