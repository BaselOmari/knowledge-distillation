class PaperParams:
    def __init__(self):
        self.epochs = 3000
        self.lr = 10
        self.momentum = 0.5
        self.distillation_weight = 0.8
    
    def optimizer_update_fn(self, optimizer, epoch):
        optimizer.param_groups[0]["lr"] *= 0.998
        if epoch < 500:
            optimizer.param_groups[0]["momentum"] += (0.99 - 0.5) / 500

class LocalParams:
    def __init__(self):
        self.epochs = 30
        self.lr = 0.1
        self.momentum = 0.5
        self.distillation_weight = 0.8
    
    def optimizer_update_fn(self, optimizer, epoch):
        optimizer.param_groups[0]["lr"] *= 0.95
