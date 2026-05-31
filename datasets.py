import torch
from torchvision import datasets, transforms

def get_mnist(batch_size=128):
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST('.', train=True, download=True, transform=transform)
    test  = datasets.MNIST('.', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=1000)
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = get_mnist()
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print(f"Sample batch shape: {next(iter(train_loader))[0].shape}, Labels shape: {next(iter(train_loader))[1].shape}")