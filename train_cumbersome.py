import matplotlib.pyplot as plt
import torch

from dataset import get_dataloader, load_mnist
from layers import CumbersomeNet
from training_params import LocalParams
from train import test, train

if __name__ == "__main__":
    train_set = load_mnist(is_train_set=True, with_jitter=True)
    train_loader = get_dataloader(train_set, batch_size=100)

    test_set = load_mnist(is_train_set=False)
    test_loader = get_dataloader(test_set, batch_size=100)

    hidden_size = 1200

    model = CumbersomeNet()

    model, loss_curve, testscores = train(model, train_loader, LocalParams(), test_loader)

    print(testscores)

    test(model, test_loader)

    torch.save(model, f"saved_models/cumbersome_model_{hidden_size}.pt")

    plt.title(f"Train vs Incorrect Count\nCumbersome Model ({hidden_size} params)")
    plt.ylabel("Incorrect")
    plt.xlabel("Epochs")
    plt.plot(testscores)
    plt.savefig(f"loss_graphs/cumbersome_{hidden_size}_incorrect.pdf", bbox_inches="tight")
    plt.show()
