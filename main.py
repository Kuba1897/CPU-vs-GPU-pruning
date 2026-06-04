import torch
import torch.optim as optim
#import copy


from models import PrunableCNN
from datasets import get_alternative
from train_test import train
from statistics import timer, count_parameters, roc_and_statistics
from pruning_utils import apply_unstructured_pruning, apply_structured_pruning

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


epochs = 2
for epoch in range(epochs):
    train(model, device, train_dataset, optimizer)


count_parameters(model, device)
timer(model, test_dataset, device)


apply_unstructured_pruning(model, True)
#count_parameters(model, device)
#timer(model, test_dataset, device)
roc_and_statistics(model, test_dataset, device, class_names)

