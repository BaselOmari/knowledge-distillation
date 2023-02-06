import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(module, mean, stddev):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=mean, std=stddev)
        if module.bias is not None:
            module.bias.data.zero_()


def clip_weights(module, max_magnitude):
    if isinstance(module, nn.Linear):
        with torch.no_grad():
            for idx in range(len(module.weight)):
                full_weights = torch.cat((module.weight[idx], torch.unsqueeze(module.bias[idx], 0)))
                if full_weights.norm() > max_magnitude:
                    normalized = F.normalize(full_weights, dim=0)
                    scaled = normalized*max_magnitude
                    module.weight.data[idx] = scaled[:-1]
                    module.bias.data[idx] = scaled[-1:]


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

        self.apply(lambda m: init_weights(m, 0, 0.01))

    def clip_weights(self, max_magnitude=15):
        self.apply(lambda m: clip_weights(m, max_magnitude))

    def forward(self, x):
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

        self.apply(lambda m: init_weights(m, 0, 0.01))

    def forward(self, x):
        x = self.inp(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits
