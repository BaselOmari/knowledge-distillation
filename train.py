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

        soft_logits = dist_logits/self.T
        soft_targets = cumb_targets/self.T

        soft_loss = -(F.softmax(soft_targets, dim=-1)*F.log_softmax(soft_logits, dim=-1)).mean()
        soft_loss *= (self.T**2)

        weighted_loss = soft_loss*self.distillation_weight + true_loss*(1 - self.distillation_weight)

        return weighted_loss



def train(model, dataset, train_params, testset):
    optimizer = optim.SGD(
        model.parameters(), lr=train_params.lr, momentum=train_params.momentum
    )
    criterion = nn.CrossEntropyLoss()

    epochLosses = []
    testScores = []

    for epoch in range(train_params.epochs):
        
        epochLoss = 0
        for input, target in tqdm(dataset):

            optimizer.zero_grad()

            logits = model(input)

            loss = criterion(logits, target)
            loss.backward()

            optimizer.step()
            # model.clip_weights()

            epochLoss += loss.item()
        
        epochLoss /= len(dataset)
        print(f"EPOCH {epoch} LOSS: {epochLoss}")
        epochLosses.append(epochLoss)

        accuracy, incorrect = test(model, testset)
        testScores.append(incorrect)

        train_params.optimizer_update_fn(optimizer, epoch)

    return model, epochLosses, testScores

def distillation(distilled_model, cumbersome_model, T, dataset, train_params, testset):
    optimizer = optim.SGD(
        distilled_model.parameters(), lr=train_params.lr, momentum=train_params.momentum
    )
    criterion = CrossEntropyLossSoft(
        train_params.distillation_weight, T
    )

    cumbersome_model.eval()

    epochLosses = []
    testScores = []

    for epoch in range(train_params.epochs):
        
        epochLoss = 0
        for input, target in tqdm(dataset):

            optimizer.zero_grad()

            distilled_logits = distilled_model(input)
            cumbersome_logits = cumbersome_model(input)

            loss = criterion(distilled_logits, cumbersome_logits, target)

            loss.backward()

            optimizer.step()

            epochLoss += loss.item()
        
        epochLoss /= len(dataset)
        print(f"EPOCH {epoch} LOSS: {epochLoss}")
        epochLosses.append(epochLoss)

        accuracy, incorrect = test(distilled_model, testset)
        testScores.append(incorrect)

        train_params.optimizer_update_fn(optimizer, epoch)

    return distilled_model, epochLosses, testScores


def test(model, test_set):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for input, target in test_set:
            logits = model(input)
            output = F.softmax(logits, dim=1)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    acc = correct/total
    incorrect = total-correct
    return acc, incorrect
