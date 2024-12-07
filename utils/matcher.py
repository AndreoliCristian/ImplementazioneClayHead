import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Abbiamo predizioni di shape [B, N, 5+num_classes]
    5 = x,y,w,h,objectness
    Usiamo un costo basato su:
    - L1 distance tra box predetti e box target
    - Class probability negativo (per favorire classi giuste)
    Per semplificare, consideriamo objectness: matchiamo solo i box GT con i box pred.
    """

    def __init__(self, cost_box=1.0, cost_class=1.0):
        super().__init__()
        self.cost_box = cost_box
        self.cost_class = cost_class

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        outputs: dict con:
          "pred_boxes": [B, N, 4]
          "pred_obj": [B, N]
          "pred_class": [B, N, num_classes]

        targets: lista lunga B di dizionari con chiavi:
          'boxes': [M, 4]
          'labels': [M]

        Ritorna una lista di B tuple (indice_pred, indice_target) per i matched.
        """
        bs = len(targets)
        indices = []

        for b in range(bs):
            pred_boxes = outputs["pred_boxes"][b]  # [N,4]
            pred_class = outputs["pred_class"][b]  # [N,num_classes]

            gt_boxes = targets[b]['boxes']  # [M,4]
            gt_labels = targets[b]['labels']  # [M]

            if gt_boxes.numel() == 0:
                # Nessun target, nessun match
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            # Calcolo costi box (L1 distance)
            # pred_boxes: [N,4], gt_boxes: [M,4]
            # Expand per broadcast:
            N = pred_boxes.size(0)
            M = gt_boxes.size(0)
            pred_boxes_exp = pred_boxes.unsqueeze(1).expand(N, M, 4)
            gt_boxes_exp = gt_boxes.unsqueeze(0).expand(N, M, 4)
            box_cost = torch.abs(pred_boxes_exp - gt_boxes_exp).sum(-1)  # [N,M]

            # Calcolo costi classi
            # estraiamo la label corretta da pred_class:
            # pred_class: [N,num_classes]
            # gt_labels: [M]
            # Cost class: -log(probabilità della classe corretta)
            pred_probs = torch.softmax(pred_class, dim=-1)  # [N,num_classes]
            class_cost = -pred_probs[:, gt_labels]  # [N,M]

            # Costo totale
            cost = self.cost_box * box_cost + self.cost_class * class_cost

            cost = cost.cpu()
            # Hungarian algorithm
            row_ind, col_ind = linear_sum_assignment(cost.numpy())
            row_ind = torch.as_tensor(row_ind, dtype=torch.long)
            col_ind = torch.as_tensor(col_ind, dtype=torch.long)

            indices.append((row_ind, col_ind))

        return indices