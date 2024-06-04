import os
import torch
from torchvision import datasets, transforms

LOCAL_RANK = int(os.environ["LOCAL_RANK"])


def setup():
    """Initialize the process group"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup():
    """Destroy the process group"""
    torch.distributed.destroy_process_group()


class CNN(torch.nn.Module):
    """Define the CNN model"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 3 * 3, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def save_checkpoint(model, epoch, path='checkpoint.pt'):
    checkpoint = {
        "model": model.module.state_dict(),
        "epochs": epoch,
    }
    torch.save(checkpoint, path)
    print(f"Epoch {epoch} | Training checkpoint saved at {path}")


def load_checkpoint(model, path='checkpoint.pt'):
    first_epoch = 0
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=f"cuda:{LOCAL_RANK}")
        model.load_state_dict(checkpoint["model"])
        first_epoch = checkpoint["epochs"]
        print(f"Resuming training from snapshot at epoch {first_epoch}")
    return model, first_epoch


def train(model, train_loader, criterion, optimizer, epoch):
    """Train the model"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(LOCAL_RANK), target.to(LOCAL_RANK)  # Move data to GPU if available
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch:"
                f" {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss:"
                f" {loss.item():.6f}"
            )


def test(model, test_loader, criterion):
    """Evaluate the model"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(LOCAL_RANK), target.to(LOCAL_RANK)  # Move data to GPU if available
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print(
        f"\nTest set: Average loss: {test_loss:.4f}, Accuracy:"
        f" {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n"
    )


def main():
    # Set up distributed training
    setup()

    # Define transformations for the training and testing sets
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    # Set datasamplers for datasets
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=LOCAL_RANK)

    # Set dataloaders for datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32000, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Create model and move it to GPU if available
    model = CNN().to(LOCAL_RANK)
    model, first_epoch = load_checkpoint(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[LOCAL_RANK])

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run training and testing
    for epoch in range(first_epoch, 50):
        train(model, train_loader, criterion, optimizer, epoch)
        test(model, test_loader, criterion)
        save_checkpoint(model, epoch)
    
    # Close distributed training
    cleanup()


if __name__ == "__main__":
    main()
