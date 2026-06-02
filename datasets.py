import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchvision import datasets

def get_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=transform)
    class_names = train.classes
    test  = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)
    return train_loader, test_loader, class_names

def get_alternative(batch_size=128):
    training_data = datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
    )

    class_names = training_data.classes
    train_dataset = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataset = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_dataset, test_dataset, class_names


if __name__ == "__main__":
    train_loader, test_loader = get_mnist()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print(f"Sample batch shape: {next(iter(train_loader))[0].shape}, Labels shape: {next(iter(train_loader))[1].shape}")