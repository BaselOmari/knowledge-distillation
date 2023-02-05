import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module, stddev):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=stddev)
        if module.bias is not None:
            module.bias.data.zero_()


def scale_vector(vector, magnitude):
    normalized = F.normalize(vector, dim=0)
    scaled = normalized * magnitude
    return scaled


def clip_weights(module, max_magnitude):
    if isinstance(module, nn.Linear):
        for idx, unit in enumerate(module.weight):
            if unit.norm() > max_magnitude:
                module.weight.data[idx] = scale_vector(unit, max_magnitude)


class CumbersomeNet(nn.Module):
    def __init__(self):
        super(CumbersomeNet, self).__init__()

        self.visible_dropout_p = 0.2
        self.hidden_dropout_p = 0.5

        # Layers
        self.inp = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.visible_dropout_p)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, 1200),
            nn.ReLU(),
            nn.Dropout(self.hidden_dropout_p)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1200, 1200),
            nn.ReLU(),
            nn.Dropout(self.hidden_dropout_p)
        )
        self.fc3 = nn.Linear(1200, 10)

        self.apply(lambda m: init_weights(m, 0.01))

    def clip_weights(self, max_magnitude=15):
        self.apply(lambda m: clip_weights(m, max_magnitude))

    def forward(self, x, temperature=1):
        x = self.inp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits
    

class SmallNet(nn.Module):
    def __init__(self, hidden_count):
        super(SmallNet, self).__init__()

        # Layers
        self.inp = nn.Sequential(
            nn.Flatten(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(28 * 28, hidden_count), nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_count, hidden_count), nn.ReLU(),
        )
        self.fc3 = nn.Linear(hidden_count, 10)

        self.apply(lambda m: init_weights(m, 0.01))

    def forward(self, x, temperature=1):
        x = self.inp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits
