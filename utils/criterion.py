import torch
import torch.nn as nn


class SetCriterion(nn.Module):
    """
    Calcola la loss tra predizioni e target dopo il matching.
    Usiamo:
    - L1 per i box matched
    - BCE per objectness
    - CrossEntropy per le classi (solo per i matched)
    """

    def __init__(self, num_classes, matcher, lambda_box=1.0, lambda_obj=1.0, lambda_class=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class

        self.l1_loss = nn.L1Loss(reduction='none')
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.ce_class = nn.CrossEntropyLoss(reduction='none')

    def forward(self, outputs, targets):
        """
        outputs: dict con:
          "pred_boxes": [B, N, 4]
          "pred_obj": [B, N]
          "pred_class": [B, N, num_classes]

        targets: lista di lunghezza B di dict:
          'boxes': [M,4]
          'labels': [M]

        Ritorna un dizionario con i valori delle loss.
        """
        # Otteniamo il matching
        indices = self.matcher(outputs, targets)

        B, N = outputs['pred_obj'].shape[:2]
        # Costruiamo maschere per matched e unmatched
        matched_idx = [(i, indices[i][0], indices[i][1]) for i in range(B)]
        # In matched_idx[i] = (batch_idx, pred_idx, tgt_idx)

        # Loss objectness:
        # Tutti i pred vengono penalizzati per objectness
        # I matched devono avere obj=1, i non matched obj=0
        obj_target = torch.zeros(B, N, device=outputs['pred_obj'].device)
        for i, (pred_inds, tgt_inds) in enumerate([(mi, ti) for _, mi, ti in matched_idx]):
            obj_target[i, pred_inds] = 1.0

        obj_loss = self.bce_obj(outputs['pred_obj'], obj_target).mean()

        # Box e class loss solo per matched
        # Concatiamo tutti i matched per batch
        src_boxes = []
        tgt_boxes = []
        src_classes = []
        tgt_classes = []

        for i, (pred_inds, tgt_inds) in enumerate([(mi, ti) for _, mi, ti in matched_idx]):
            if len(pred_inds) > 0:
                src_boxes.append(outputs['pred_boxes'][i, pred_inds])
                tgt_boxes.append(targets[i]['boxes'][tgt_inds])

                src_classes.append(outputs['pred_class'][i, pred_inds])
                tgt_classes.append(targets[i]['labels'][tgt_inds])

        if len(src_boxes) > 0:
            src_boxes = torch.cat(src_boxes, dim=0)
            tgt_boxes = torch.cat(tgt_boxes, dim=0)
            box_loss = self.l1_loss(src_boxes, tgt_boxes).mean()

            src_classes = torch.cat(src_classes, dim=0)
            tgt_classes = torch.cat(tgt_classes, dim=0)
            class_loss = self.ce_class(src_classes, tgt_classes).mean()
        else:
            # Nessun match
            box_loss = torch.tensor(0.0, device=outputs['pred_obj'].device)
            class_loss = torch.tensor(0.0, device=outputs['pred_obj'].device)

        loss = self.lambda_box * box_loss + self.lambda_obj * obj_loss + self.lambda_class * class_loss
        loss_dict = {
            "loss": loss,
            "loss_box": box_loss,
            "loss_obj": obj_loss,
            "loss_class": class_loss
        }
        return loss_dict