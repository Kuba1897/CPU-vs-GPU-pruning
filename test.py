import torch
import torch.optim as optim
#import copy


from models import PrunableCNN
from datasets import get_alternative
from train_test import train
from statistics import timer, count_parameters, roc_and_statistics
from pruning_utils import apply_unstructured_pruning, apply_structured_pruning




def structured_pruning_test_gpu():
    batch_size = 128
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA not available, aborting.")
        return

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    apply_structured_pruning(model, device, regularization=2, amount=0.2)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    recovery_epochs = 5
    for epoch in range(recovery_epochs):
        train(model, device, train_dataset, optimizer)

    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_structured_pruned_gpu.png")


def unstructured_pruning_test_gpu():
    batch_size = 128
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA not available, aborting.")
        return

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    apply_unstructured_pruning(model, False)
    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_unstructured_pruned_gpu.png")


def structured_pruning_test_cpu():
    batch_size = 128
    device = "cpu"

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    apply_structured_pruning(model, device, regularization=2, amount=0.2)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    recovery_epochs = 5
    for epoch in range(recovery_epochs):
        train(model, device, train_dataset, optimizer)

    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_structured_pruned_cpu.png")


def unstructured_pruning_test_cpu():
    batch_size = 128
    device = "cpu"

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    apply_unstructured_pruning(model, False)
    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_unstructured_pruned_cpu.png")


def non_pruning_test_gpu():
    batch_size = 128
    if torch.cuda.is_available():
        device = "cuda"
    else:
        print("CUDA not available, aborting.")
        return

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_non_pruned_gpu.png")


def non_pruning_test_cpu():
    batch_size = 128
    device = "cpu"

    train_dataset, test_dataset, class_names = get_alternative()

    model = PrunableCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    for epoch in range(epochs):
        train(model, device, train_dataset, optimizer)

    count_parameters(model, device)
    timer(model, test_dataset, device)
    roc_and_statistics(model, test_dataset, device, class_names, "results/roc_non_pruned_cpu.png")


if __name__ == "__main__":
    print("Running non-pruning test on GPU...")
    non_pruning_test_gpu()

    print("\nRunning unstructured pruning test on GPU...")
    unstructured_pruning_test_gpu()

    print("\nRunning structured pruning test on GPU...")
    structured_pruning_test_gpu()

    print("\nRunning non-pruning test on CPU...")
    non_pruning_test_cpu()

    print("\nRunning unstructured pruning test on CPU...")
    unstructured_pruning_test_cpu()

    print("\nRunning structured pruning test on CPU...")
    structured_pruning_test_cpu()