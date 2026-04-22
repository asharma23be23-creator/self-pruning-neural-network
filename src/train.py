import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from model import PrunableNet
from utils import sparsity_loss, compute_sparsity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAMBDA = 0.0001

transform = transforms.ToTensor()

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=transform),
    batch_size=1000)

model = PrunableNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss += LAMBDA * sparsity_loss(model)

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

# Evaluation
model.eval()
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        pred = model(data).argmax(dim=1)
        correct += pred.eq(target).sum().item()

accuracy = 100. * correct / len(test_loader.dataset)
sparsity = compute_sparsity(model)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Sparsity: {sparsity:.2f}%")
