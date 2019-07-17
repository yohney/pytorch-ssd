import torch.nn as nn
import torch.nn.functional as F
import torch

from ..utils import box_utils

class NearMissLoss(nn.Module):
    def __init__(self, neg_pos_ratio, num_classes, device):
        super(NearMissLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio
        self.num_classes = num_classes
        self.cew = torch.ones([num_classes], dtype=torch.float32)
        self.cew[1:] = 2
        self.cew.to(device)

    def get_bg_nearmiss_mask(self, conf, conf_pred, cnt):
        true_neg_mask = conf == 0
        pred_bg_mask = conf_pred.max(2)[1] == 0

        bg_confs = conf_pred[:,:,0]
        non_bg_confs = conf_pred[:,:,1:].max(2)[0]
        diffs = (bg_confs - non_bg_confs).abs()

        thr = diffs[true_neg_mask & pred_bg_mask].topk(cnt, sorted=False, largest=False)[0].max()

        diff_mask = ((diffs < thr) & true_neg_mask & pred_bg_mask)
        
        return diff_mask

    def forward(self, conf_pred, loc_pred, conf, loc):
        pos_mask = conf>0
        
        pred_pos_mask = conf_pred.max(2)[1] > 0
        nm_mask = self.get_bg_nearmiss_mask(conf, conf_pred, int(pos_mask.sum()) * self.neg_pos_ratio)
        all_mask = (pos_mask + pred_pos_mask + nm_mask).clamp(0, 1)

        assert(conf[~all_mask].sum() == 0) # validate masking algorithm
        conf_loss = F.cross_entropy(conf_pred[all_mask].view(-1, self.num_classes), conf[all_mask].view(-1), reduction='sum', weight=self.cew)
        num_matched_default_boxes = float(all_mask.data.long().sum())
        conf_loss /= num_matched_default_boxes

        # localization loss (samo za matchane groundtruth bboxeve)
        pos_mask = pos_mask.unsqueeze(2).expand_as(loc_pred)
        pos_loc_preds = loc_pred[pos_mask].view(-1,4)
        pos_loc_targets = loc[pos_mask].view(-1,4)

        loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, reduction='sum')
        loc_loss /= num_matched_default_boxes

        self.conf_loss = conf_loss
        self.loc_loss = loc_loss

        return loc_loss, conf_loss
