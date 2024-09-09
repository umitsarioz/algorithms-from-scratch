import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Convolutional Layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Convolutional Layer 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128) # Fully Connected Layer 1
        self.fc2 = nn.Linear(128, 10) # Output Layer

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) # Convolutional + ReLU + Pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2)) # Convolutional + ReLU + Pooling
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x)) # Fully Connected + ReLU
        x = self.fc2(x) # Output Layer
        return F.log_softmax(x, dim=1) # Softmax for classification

# Hyperparameters and data preparation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=64, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, download=True, transform=transform), batch_size=1000, shuffle=False)

model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train(model, device, train_loader, optimizer, epoch):
    print("train started")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Testing and evaluating function
def test(model, device, test_loader):
    print("test started")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

# Training and evaluating the model
for epoch in range(1, 5):
    print(epoch,". epoch started")
    train(model, device, train_loader, optimizer, epoch)
accuracy = test(model, device, test_loader)
