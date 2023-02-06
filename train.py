# Imports
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from dataset import get_dataloader, load_mnist
from layers import CumbersomeNet
from training_params import LocalParams
from utils import CrossEntropyLossSoft, test


def train(model, dataset, testset, train_params):
    optimizer = optim.SGD(
        model.parameters(), lr=train_params.lr, momentum=train_params.momentum
    )
    criterion = nn.CrossEntropyLoss()

    validation_scores = []
    epoch_losses = []
    for epoch in range(train_params.epochs):
        # Set to Training Mode
        model.train()

        epoch_loss = 0
        for input, target in tqdm(dataset):
            optimizer.zero_grad()

            logits = model(input)

            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            # Comment out if not training cumbersome model
            model.clip_weights()

            epoch_loss += loss.item()

        epoch_loss /= len(dataset)
        epoch_losses.append(epoch_loss)

        acc, incorrect = test(model, testset)
        validation_scores.append(acc)

        print(f"Epoch {epoch} loss: {epoch_loss}")
        print(f"Epoch {epoch} acc: {acc}")

        # update optimizer hyperparameters after every epoch
        train_params.optimizer_update_fn(optimizer, epoch)

    return model, epoch_loss, validation_scores


if __name__ == "__main__":
    # Load Training Hyperparameters
    params = LocalParams()

    # Load Datasets
    train_set = load_mnist(is_train_set=True, with_jitter=True)
    train_loader = get_dataloader(train_set, batch_size=params.batch_size)

    test_set = load_mnist(is_train_set=False)
    test_loader = get_dataloader(test_set)

    # Create Model
    model = CumbersomeNet()

    # Run Distillation
    model, losses, validation_scores = train(
        model=model,
        dataset=train_loader,
        testset=test_loader,
        train_params=params
    )

    # Plot Results
    plt.title(f"Validation Set Accuracy\nCumbersome Model (1200 Units)")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(validation_scores)
    plt.show()
