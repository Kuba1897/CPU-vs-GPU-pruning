import torch
import torch.nn as nn

def train(model, device, train_loader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f"Epoch done.")


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset)
    return test_loss, acc