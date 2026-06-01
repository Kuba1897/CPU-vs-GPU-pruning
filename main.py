import torch
import torch.optim as optim


from models import PrunableCNN
from datasets import get_alternative
from train_test import train, test
from statistics import accuracy, count_parameters, roc_and_statistics
from pruning_utils import apply_unstructured_pruning

batch_size = 128
device = "cpu"

print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    device = "cuda"
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())

train_dataset, test_dataset, class_names = get_alternative()


model = PrunableCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


epochs = 5

for epoch in range(epochs):
    train(
        model, device, train_dataset, optimizer
    )



test_acc, tmer = accuracy(model, test_dataset, device)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Approximate time needed for forward pass of 1 batch of inputs(120) using gpu: {tmer} seconds")


total_params, trainable_params, _ , _ = count_parameters(model)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")


roc_and_statistics(model, test_dataset, device, class_names)


apply_unstructured_pruning(model)

total_params, trainable_params, no_zero , sparce = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Parameters with value 0: {total_params-no_zero}")
print(f"Sparcity:  {sparce}")

for epoch in range(1):
    train(
        model, device, train_dataset, optimizer
    )

test_acc, tmer = accuracy(model, test_dataset, device)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Approximate time needed for forward pass of 1 batch of inputs(120) using gpu: {tmer} seconds")