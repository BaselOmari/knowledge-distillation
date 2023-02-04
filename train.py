import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm


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

            output, logits = model(input)

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
            output, logit = model(input)
            correct += (output.argmax(1) == target).sum().item()
            total += target.size(0)
    print("Incorrect Count:", total-correct)
    return correct / total
