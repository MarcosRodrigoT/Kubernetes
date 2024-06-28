"""
To launch multi-worker training in PyTorch, you should:

    1. Clone your code on all the PCs (nodes).
    
    2. Run the following command on all the nodes:
    
        torchrun \
            --nproc-per-node=<num-gpus> \
            --nnodes=<num-nodes> \
            --node-rank=0 \
            --rdzv-endpoint=<ip-address>:<port> \
            multi_worker_plus.py  # Put here additional arguments to your script (if any): --arg1 --arg2...
            
        where...
            -nproc-per-node: number of GPUs on the current PC. You may have PCs with different number of GPUs.
            -nnodes: number of nodes (PCs).
            -node-rank: id for the current node. Use 0 for the master node (choose any PC to become master) and 1, 2, ... for the rest.
            -rdzv-endpoint: ip address (or DNS name) and port (default is 29400) of master node. Only for multi-node training.
            
    3. If connection fails, NCCL might have failed in automatically searching the network interface. Use 'ifconfig' to find the correct
       network interface (probably 'eth0' or 'eno1') and run the following command:
    
            export NCCL_SOCKET_IFNAME=<net-interface>
            
Additional info: https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html
"""

import os
import torch
import pickle
import numpy as np
from torchvision import datasets, transforms

LOCAL_RANK = int(os.environ["LOCAL_RANK"])

def setup():
    """Initialize the process group"""
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(LOCAL_RANK)

def cleanup():
    """Destroy the process group"""
    torch.distributed.destroy_process_group()

class CIFAR10Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.train = train
        
        # Load train or test data
        if self.train:
            self.data, self.labels = self.load_data('data_batch_')
        else:
            self.data, self.labels = self.load_data('test_batch')
        
    def load_data(self, file_prefix):
        data = []
        labels = []
        if self.train:
            for i in range(1, 6):
                file_path = os.path.join(self.data_dir, f'{file_prefix}{i}')
                with open(file_path, 'rb') as f:
                    batch = pickle.load(f, encoding='bytes')
                    data.append(batch[b'data'])
                    labels.extend(batch[b'labels'])
            data = np.concatenate(data)
        else:
            file_path = os.path.join(self.data_dir, file_prefix)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
                data = batch[b'data']
                labels = batch[b'labels']
        
        # Reshape and transpose to get the correct format
        data = data.reshape(-1, 32, 32, 3).astype(np.float32)
        return data, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        image = self.data[index]
        label = self.labels[index]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class CNN(torch.nn.Module):
    """Define the CNN model"""
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 64)
        self.fc2 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR10 dataset. First you should:
    #   1. Download the dataset from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
    #   2. Uncompress it with: tar -xvzf cifar-10-python.tar.gz
    #   3. Place the files on 'data/cifar10'
    train_dataset = CIFAR10Dataset(data_dir='./data/cifar10', train=True, transform=transform)
    test_dataset = CIFAR10Dataset(data_dir='./data/cifar10', train=False, transform=transform)
    
    # Set datasamplers for datasets
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=LOCAL_RANK)

    # Set dataloaders for datasets
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=3200, sampler=train_sampler)
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
