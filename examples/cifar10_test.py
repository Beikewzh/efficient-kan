import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np
import random
# Assuming KAN is correctly imported and initialized
from efficient_kan import KAN
from torchvision import models

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

class MLP(nn.Module):
    def __init__(self, classes = 10):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, classes)
        )

    def forward(self, x):
        return self.layers(x)


class CNN(nn.Module):
    def __init__(self, classes = 10):
        super(CNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.LayerNorm([16, 16, 16]),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.LayerNorm([32, 8, 8]),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 10)
            # KAN([32 * 4 * 4, 64, 10])
        )

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        return self.layers(x)
# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation
def one_hot_encode(labels, n_classes=5):
    return torch.eye(n_classes)[labels].to(device)

def custom_cross_entropy(predictions, targets):
    # Applying Log Softmax to get log probabilities
    log_softmax = nn.LogSoftmax(dim=1)
    log_probs = log_softmax(predictions)
    # Cross-entropy loss calculation
    loss = -torch.sum(targets * log_probs) / targets.size(0)
    return loss

def compute_fisher(model, input_size, dataloader, criterion, device):
    fisher_info = {}
    model.eval()
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param.data)
    
    # Use the model's output as labels to compute log likelihood
    for images, labels in dataloader:
        images = images.view(-1, input_size).to(device)
        outputs = model(images)
        labels = outputs.max(1)[1]  # Use the model's predictions as labels
        model.zero_grad()
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher_info[name] += param.grad.data ** 2 / len(dataloader.dataset)
    
    return fisher_info

def ewc_loss(model, fisher, opt_params, lambda_ewc):
    loss = 0
    for name, param in model.named_parameters():
        loss += (fisher[name] * (opt_params[name] - param).pow(2)).sum()    
    return lambda_ewc * loss

# Splitting the dataset into two tasks
def split_dataset(task_num, num_classes):
    class_indices = list(range(num_classes))
    random.shuffle(class_indices)
    task_size = num_classes // task_num
    phase_classes = [set(class_indices[i * task_size:(i + 1) * task_size]) for i in range(task_num)]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    split_train_datset = []
    split_test_datset = []
    print(f'class in different tasks {phase_classes}')
    for classes in phase_classes:
        indices = [i for i, label in enumerate(train_dataset.targets) if label in classes]
        test_indices = [i for i, label in enumerate(test_dataset.targets) if label in classes]
        subset = Subset(train_dataset, indices)
        test_subset = Subset(test_dataset, test_indices)
        split_train_datset.append(subset)
        split_test_datset.append(test_subset)
    return train_dataset, test_dataset, split_train_datset, split_test_datset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Model initialization
model = KAN([3 * 32 * 32, 64, 10]).to(device)
# model = CNN().to(device)


for idx, layer in enumerate(model.layers):
    print(f"Layer {idx}: {layer.__class__.__name__}")
    for name, param in layer.named_parameters():
        print(f"  {name} shape: {param.shape}")

# Specifically print details about the last layer
last_layer = model.layers[-1]
print("\nDetails of the Last Layer:")
print(last_layer)
for name, param in last_layer.named_parameters():
    print(f"  {name} shape: {param.shape}")

print(f'total number of parameters {count_parameters(model)}')



# Training loop
def train_task(task_num, input_size, train_loader, test_loader, optimizer, fisher=None, opt_params=None, lambda_ewc=0, epochs=5):
    for epoch in range(epochs):  # Train each task for 5 epochs
        model.train()
        train_loss, train_accuracy = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.view(-1, input_size).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if fisher and opt_params:
                loss += ewc_loss(model, fisher, opt_params, lambda_ewc)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (outputs.argmax(1) == labels).float().mean().item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        print(f'Task {task_num}, Epoch {epoch+1}: Train Loss {train_loss:.4f}, Train Accuracy {train_accuracy:.4f}')

        # Evaluate on test data
        model.eval()
        test_loss, test_accuracy = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.view(-1, input_size).to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_accuracy += (outputs.argmax(1) == labels).float().mean().item()

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        print(f'Task {task_num}, Epoch {epoch+1}: Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy:.4f}')
    print("--------------------------------task done--------------------------------")


TASK_NUM = 2
INPUT_SIZE = 32
INPUT_DIM = 3 * 32 * 32
train_dataset, test_dataset, split_train_datset, split_test_datset = split_dataset(TASK_NUM, 10)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# TASK 1
train_loader1 = DataLoader(split_train_datset[0], batch_size=64, shuffle=True)
test_loader1 = DataLoader(split_test_datset[0], batch_size=64, shuffle=False)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
train_task(1, INPUT_DIM, train_loader1, test_loader1, optimizer, epochs=5)

opt_params = {n: p.clone().detach() for n, p in model.named_parameters()}
fisher = compute_fisher(model, INPUT_DIM, test_loader, criterion, device)
lambda_ewc = 2500
# TASK2
# for param in model.parameters():
#     param.requires_grad = True

# for param in model.layers[-1].parameters():
#     param.requires_grad = False

# optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
optimizer = optim.AdamW(model.parameters(), lr=4e-5, weight_decay=2e-6)

train_loader2 = DataLoader(split_train_datset[1], batch_size=64, shuffle=True)
test_loader2 = DataLoader(split_test_datset[1], batch_size=64, shuffle=False)
train_task(2, INPUT_DIM, train_loader2, test_loader2, optimizer, fisher=fisher, opt_params=opt_params, lambda_ewc=lambda_ewc, epochs=5)


# TASK 1 ACC
model.eval()
total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader1):
        images, labels = images.view(-1, INPUT_DIM).to(device), labels
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'TASK 1 Test Accuracy: {total_accuracy / len(test_loader1):.4f}')

# TASK 2 ACC
total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader2):
        images, labels = images.view(-1, INPUT_DIM).to(device), labels
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'TASK 2 Test Accuracy: {total_accuracy / len(test_loader2):.4f}')

# OVERALL ACC
total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images, labels = images.view(-1, INPUT_DIM).to(device), labels
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'Overall Test Accuracy: {total_accuracy / len(test_loader):.4f}')