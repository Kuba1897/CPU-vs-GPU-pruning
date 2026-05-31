import torch
from models import SimpleCNN
from datasets import get_mnist
from pruning_utils import apply_unstructured_pruning, apply_structured_pruning
from train_test import train, test

device_cpu = torch.device("cpu")
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(model_name, apply_prune, device):
    train_loader, test_loader = get_mnist()
    model = SimpleCNN().to(device)
    if apply_prune == "unstructured":
        model = apply_unstructured_pruning(model, amount=0.5)
    elif apply_prune == "structured":
        model = apply_structured_pruning(model, amount=0.5)
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(3):
        train(model, device, train_loader, optimizer, epoch)
    test_loss, acc = test(model, device, test_loader)
    print(f"{model_name} | {apply_prune} | {device}: Test Loss: {test_loss:.4f}, Acc: {acc:.4f}")

configs = [
    ('no_pruning', None),
    ('unstructured', "unstructured"),
    ('structured', "structured"),
]
devices = [device_cpu, device_gpu] if torch.cuda.is_available() else [device_cpu]

for device in devices:
    for name, prune_type in configs:
        run(name, prune_type, device)