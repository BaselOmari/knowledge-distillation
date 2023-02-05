class PaperParams:
    """
    Parameters as specified in the paper "Improving neural networks by preventing
    co-adaptation of feature detectors" by Hinton et al.
    """
    def __init__(self):
        self.epochs = 3000
        self.lr = 10
        self.momentum = 0.5
        self.distillation_weight = 0.8
        self.batch_size = 100
    
    def optimizer_update_fn(self, optimizer, epoch):
        optimizer.param_groups[0]["lr"] *= 0.998
        if epoch < 500:
            optimizer.param_groups[0]["momentum"] += (0.99 - 0.5) / 500

class LocalParams:
    """
    Reduced parameters used for local demonstration, adjusted to meet computational
    and time capacity
    """
    def __init__(self):
        self.epochs = 30
        self.lr = 0.1
        self.momentum = 0.5
        self.distillation_weight = 0.8
        self.batch_size = 100
    
    def optimizer_update_fn(self, optimizer, epoch):
        optimizer.param_groups[0]["lr"] *= 0.95
