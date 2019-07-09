class LayerDescriptor(object):
    def __init__(self, idx, shape, num_anchors, num_classes):
        self.idx = idx
        self.shape = shape
        self.num_anchors = num_anchors
        self.num_classes = num_classes

    def header(self):
        return "{} ({}x{})".format(self.idx, self.shape[0], self.shape[1])