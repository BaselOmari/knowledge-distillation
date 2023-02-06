import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


class CrossEntropyLossSoft(nn.Module):
    """
    Objective function of distilled model which combines:
    - Cross-Entropy Loss with True Labels
    - Cross-Entropy Loss with Soft Targets
    """

    def __init__(self, weight, temperature):
        super(CrossEntropyLossSoft, self).__init__()
        self.distillation_weight = weight
        self.T = temperature

    def forward(self, dist_logits, cumb_targets, true_targets):
        true_loss = F.cross_entropy(dist_logits, true_targets)

        soft_logits = dist_logits / self.T
        soft_targets = cumb_targets / self.T

        soft_loss = -1 * (F.softmax(soft_targets, dim=-1) * F.log_softmax(soft_logits, dim=-1)).mean()
        soft_loss *= self.T**2

        weighted_loss = soft_loss * self.distillation_weight + true_loss * (
            1 - self.distillation_weight
        )

        return weighted_loss


def test(model, test_set):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in test_set:
            logits = model(input)
            output = F.softmax(logits, dim=1)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    acc = correct / total
    incorrect = total - correct
    return acc, incorrect
