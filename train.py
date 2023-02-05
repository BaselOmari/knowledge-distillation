import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm


class CrossEntropyLossSoft(nn.Module):
    def __init__(self, weight, temperature):
        super(CrossEntropyLossSoft, self).__init__()
        self.distillation_weight = weight
        self.T = temperature

    def forward(self, d_logits, c_targets, true_targets):

        true_loss = F.cross_entropy(d_logits, true_targets)

        soft_logits = d_logits/self.T
        soft_targets = c_targets/self.T

        soft_loss = F.cross_entropy(soft_logits, soft_targets)*(self.T**2)

        weighted_loss = soft_loss*self.distillation_weight + true_loss*(1 - self.distillation_weight)

        return weighted_loss



def train(model, dataset, train_params):
    optimizer = optim.SGD(
        model.parameters(), lr=train_params.lr, momentum=train_params.momentum
    )
    criterion = nn.CrossEntropyLoss()

    epochLosses = []

    for epoch in range(train_params.epochs):
        
        epochLoss = 0
        for input, target in tqdm(dataset):

            optimizer.zero_grad()

            logits = model(input)

            loss = criterion(logits, target)
            loss.backward()

            optimizer.step()
            model.clip_weights()

            epochLoss += loss.item()
        
        epochLoss /= len(dataset)
        print(f"EPOCH {epoch} LOSS: {epochLoss}")
        epochLosses.append(epochLoss)

        train_params.optimizer_update_fn(optimizer, epoch)

    return model, epochLosses

def test(model, test_set):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in test_set:
            logits = model(input)
            output = F.softmax(logits, dim=1)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    print("Incorrect Count:", total-correct)
    return correct / total
