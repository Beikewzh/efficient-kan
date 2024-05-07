import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np

# Assuming KAN is correctly imported and initialized
from efficient_kan import KAN
torch.manual_seed(0)

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

# Setting up the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def one_hot_encode(labels, n_classes=5):
    return torch.eye(n_classes)[labels].to(device)

def custom_cross_entropy(predictions, targets):
    # Applying Log Softmax to get log probabilities
    log_softmax = nn.LogSoftmax(dim=1)
    log_probs = log_softmax(predictions)
    # Cross-entropy loss calculation
    loss = -torch.sum(targets * log_probs) / targets.size(0)
    return loss

# Splitting the dataset into two tasks
def split_mnist(task_num):
    classes = [0, 1, 2, 3, 4] if task_num == 1 else [5, 6, 7, 8, 9]

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create masks for training and testing datasets
    train_mask = np.isin(train_dataset.targets.numpy(), classes)
    test_mask = np.isin(test_dataset.targets.numpy(), classes)

    # Apply masks
    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = train_dataset.targets[train_mask]

    test_dataset.data = test_dataset.data[test_mask]
    test_dataset.targets = test_dataset.targets[test_mask]

    unique_train_classes = np.unique(train_dataset.targets.numpy())
    unique_test_classes = np.unique(test_dataset.targets.numpy())
    print(f"Unique classes in train dataset: {unique_train_classes}")
    print(f"Unique classes in test dataset: {unique_test_classes}")

    #train_dataset.targets = torch.tensor(train_dataset.targets) - classes[0]
    #test_dataset.targets = torch.tensor(test_dataset.targets) - classes[0]

    return train_dataset, test_dataset


# Model initialization
model = KAN([28 * 28, 64, 10]).to(device)
#model = MLP(classes=10).to(device)

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


# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# Compute Fisher Information Matrix
def compute_fisher(model, dataloader, device):
    fisher_info = {}
    model.eval()
    for name, param in model.named_parameters():
        fisher_info[name] = torch.zeros_like(param.data)
    
    # Use the model's output as labels to compute log likelihood
    for images, labels in dataloader:
        images = images.view(-1, 28 * 28).to(device)
        outputs = model(images)
        labels = outputs.max(1)[1]  # Use the model's predictions as labels
        model.zero_grad()
        outputs = torch.nn.functional.log_softmax(outputs, dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        
        for name, param in model.named_parameters():
            fisher_info[name] += param.grad.data ** 2 / len(dataloader)
    
    return fisher_info

# EWC Loss
def ewc_loss(model, fisher, opt_params, lambda_ewc):
    loss = 0
    for name, param in model.named_parameters():
        loss += (fisher[name] * (opt_params[name] - param).pow(2)).sum()    
    return lambda_ewc * loss


# Training loop
def train_task(task_num, train_dataset, test_dataset, epochs=5,fisher=None, opt_params=None, lambda_ewc=0):
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for epoch in range(epochs):  # Train each task for 5 epochs
        model.train()
        train_loss, train_accuracy = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
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
                images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_accuracy += (outputs.argmax(1) == labels).float().mean().item()

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        print(f'Task {task_num}, Epoch {epoch+1}: Test Loss {test_loss:.4f}, Test Accuracy {test_accuracy:.4f}')
    print("--------------------------------task done--------------------------------")



train_dataset_1, test_dataset_1 = split_mnist(1)
# Execute training for both tasks
train_task(1, train_dataset_1, test_dataset_1, epochs=10)
# for param in model.parameters():
#     param.requires_grad = False

# for param in model.layers[-1].parameters():
#     param.requires_grad = True

opt_params = {n: p.clone().detach() for n, p in model.named_parameters()}
fisher = compute_fisher(model, test_dataset_1, device)

# Task 2: Train on Fashion-MNIST with EWC
lambda_ewc = 500  # Regularization strength for EWC
optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

train_dataset_2, test_dataset_2 = split_mnist(2)

train_task(2, train_dataset_2, test_dataset_2, epochs=10, fisher=fisher, opt_params=opt_params, lambda_ewc=lambda_ewc)


model.eval()
total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(test_dataset_1):
        images, labels = images.view(-1, 28 * 28).to(device), labels
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'Task 1 Accuracy: {total_accuracy / len(test_dataset_1):.4f}')

total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(test_dataset_2):
        images, labels = images.view(-1, 28 * 28).to(device), labels
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'Task 2 Accuracy: {total_accuracy / len(test_dataset_2):.4f}')