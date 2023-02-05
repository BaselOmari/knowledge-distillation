import matplotlib.pyplot as plt
import torch

from dataset import get_dataloader, load_mnist
from layers import SmallNet
from training_params import LocalParams
from train import test, distillation

if __name__ == "__main__":
    train_set = load_mnist(is_train_set=True, with_jitter=True)
    train_loader = get_dataloader(train_set, batch_size=100)

    test_set = load_mnist(is_train_set=False)
    test_loader = get_dataloader(test_set, batch_size=100)

    hidden_size = 800
    temperature = 20

    distilled_model = SmallNet(hidden_size)

    cumbersome_model = torch.load("saved_models\cumbersome_model_30.pt")

    model, loss_curve, testscores = distillation(distilled_model, cumbersome_model, temperature, train_loader, LocalParams(), test_loader)

    print(testscores)

    test(model, test_loader)

    torch.save(model, f"saved_models/distilled_model_{hidden_size}_20.pt")

    plt.title(f"Train vs Incorrect Count\nDistilled Model ({hidden_size} params; T={temperature})")
    plt.ylabel("Incorrect")
    plt.xlabel("Epochs")
    plt.plot(testscores)
    plt.savefig(f"loss_graphs/distilled_{hidden_size}_{temperature}_incorrect.pdf", bbox_inches="tight")
    plt.show()
