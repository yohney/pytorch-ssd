from ..utils.misc import Timer
import torch

class LayerDescriptor(object):
    def __init__(self, idx, shape, num_anchors, num_classes):
        self.idx = idx
        self.shape = shape
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.priors = None

    def get_priors(self):
        if self.priors is None:
            self.priors = self._calculate_priors()

        return self.priors

    def get_box_count(self):
        return self.num_anchors * self.shape[0] * self.shape[1]

    def header(self):
        return "{} ({}x{})".format(self.idx, self.shape[0], self.shape[1])

    def build_stats(self):
        return LayerStatistics(self)

class LayerApCalculator(object):
    def __init__(self, net):
        self.timer = Timer()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net

    def update_layer_stats(self, layer_stats, images, gt_confidences):
        if len(images.shape) == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device)
        gt_confidences = gt_confidences.to(self.device)

        with torch.no_grad():
            self.timer.start()
            confidences, _ = self.net.forward(images)
            self.timer.end()
            #print("Inference time: ", self.timer.end())

        confidences = torch.argmax(confidences[0], dim=-1)
        pos_mask = gt_confidences > 0
        bg_mask = gt_confidences == 0

        start = 0
        end = 0
        for ls in layer_stats:
            end += int(ls.descriptor.get_box_count())

            c_pred = confidences[start:end]
            c_gt = gt_confidences[start:end]

            c_pos_mask = pos_mask[start:end]
            c_bg_mask = bg_mask[start:end]

            ls.non_bg_boxes += torch.clamp(c_gt[c_pos_mask], max=1).sum().cpu().item()
            ls.correct += torch.eq(c_pred[c_pos_mask], c_gt[c_pos_mask]).sum().cpu().item()
            ls.miss_class += torch.ne(c_pred[c_pos_mask], c_gt[c_pos_mask]).sum().cpu().item() + (torch.clamp(c_pred[c_pos_mask], max=1) - 1).sum().cpu().item()
            ls.fpos += torch.clamp(c_pred[c_bg_mask], max=1).sum().cpu().item()

            start = end

class LayerStatistics(object):
    def __init__(self, layer_descriptor):
        self.descriptor = layer_descriptor

        self.non_bg_boxes = 0
        self.correct = 0
        self.miss_class = 0
        self.fpos = 0