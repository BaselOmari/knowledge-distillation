# Imports
import matplotlib.pyplot as plt
import torch
from torch import optim
from tqdm import tqdm

from dataset import get_dataloader, load_mnist, filter_by_class
from layers import SmallNet
from training_params import LocalParams
from utils import CrossEntropyLossSoft, test


def distill(distilled_model, cumbersome_model, T, dataset, testset, three_testset, train_params):
    optimizer = optim.SGD(
        distilled_model.parameters(), lr=train_params.lr, momentum=train_params.momentum
    )
    criterion = CrossEntropyLossSoft(
        train_params.distillation_weight, T
    )

    # Set to Inference Mode (Disable Dropout)
    cumbersome_model.eval()

    validation_scores = []
    three_scores = []
    epoch_losses = []
    for epoch in range(train_params.epochs):

        # Set to Training Mode
        distilled_model.train()

        epoch_loss = 0
        for input, target in tqdm(dataset):
            optimizer.zero_grad()

            distilled_logits = distilled_model(input)
            cumbersome_logits = cumbersome_model(input)

            loss = criterion(distilled_logits, cumbersome_logits, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        epoch_loss /= len(dataset)
        epoch_losses.append(epoch_loss)
    
        acc_total, incorrect_total = test(distilled_model, testset)
        validation_scores.append(acc_total)

        acc_three, incorrect_three = test(distilled_model, three_testset)
        three_scores.append(acc_three)

        print(f"Epoch {epoch} loss: {epoch_loss}")
        print(f"Epoch {epoch} acc: {acc_total}")
        print(f"Epoch {epoch} three acc: {acc_three}")

        # update optimizer hyperparameters after every epoch
        train_params.optimizer_update_fn(optimizer, epoch)
    
    return distilled_model, epoch_losses, validation_scores, three_scores



if __name__=="__main__":
    # Load Training Hyperparameters
    params = LocalParams()

    # Load Train Set
    classes = [i for i in range(9+1)]
    classes.remove(3)

    train_set = load_mnist(is_train_set=True, with_jitter=True)
    filtered_set = filter_by_class(train_set, classes)
    train_loader = get_dataloader(filtered_set, batch_size=params.batch_size)

    # Load Test Sets
    test_set = load_mnist(is_train_set=False)
    test_loader = get_dataloader(test_set)

    only_three_set = filter_by_class(test_set, [3])
    only_three_loader = get_dataloader(only_three_set)

    # Define Hidden Size and Temperature
    hidden_size = 300
    temperature = 8

    # Load Cumbersome model and Create Small Network
    cumbersome_model = torch.load("models\cumbersome_model_1200.pt")
    distilled_model = SmallNet(hidden_size)

    # Run Distillation
    distilled_model, losses, validation_scores, three_scores = distill(
        distilled_model=distilled_model,
        cumbersome_model=cumbersome_model,
        T=temperature,
        dataset=train_loader,
        testset=test_loader,
        three_testset=only_three_loader,
        train_params=params
    )

    # Test with increased digit 3 learned bias
    distilled_model.fc3.bias.data[3] += 3.5

    acc_total, incorrect_total = test(distilled_model, test_loader)
    acc_three, incorrect_three = test(distilled_model, only_three_loader)

    print("Increased Digit 3 Learned Bias by +3.5")
    print(f"acc: {acc_total}")
    print(f"three acc: {acc_three}")

    # Plot Results
    plt.title(f"Validation Set Accuracy\nDistilled Model Trained without Digit 3 ({hidden_size} units; T={temperature})")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(validation_scores, label='All Classes')
    plt.plot(three_scores, label='Digit 3')
    plt.legend()
    plt.show()
