import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from tqdm import tqdm
import numpy as np

# Assuming KAN is correctly imported and initialized
from efficient_kan import KAN

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
#########################################
#########################################
#########################################

class FashionMNISTAdjusted(Dataset):
    def __init__(self, root, train, download, transform):
        self.dataset = datasets.FashionMNIST(root=root, train=train, download=download, transform=transform)
    
    def __getitem__(self, index):
        image, label = self.dataset[index]
        label += 10  # Adjust label to be in the range 10 to 19
        return image, label
    
    def __len__(self):
        return len(self.dataset)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion-MNIST training data with adjusted labels
trainset = FashionMNISTAdjusted(
    root='./data', 
    train=True, 
    download=True, 
    transform=transform
)

# Load Fashion-MNIST validation data with adjusted labels
valset = FashionMNISTAdjusted(
    root='./data', 
    train=False, 
    download=True, 
    transform=transform
)

# Create data loaders
fashion_trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
fashion_valloader = DataLoader(valset, batch_size=64, shuffle=False)

def get_all_labels(dataloader):
    all_labels = []
    for _, labels in dataloader:
        all_labels.extend(labels.tolist())  # Append labels to the list
    return all_labels


#########################################
#########################################
#########################################

trainset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
valset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
mnist_trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
mnist_valloader = DataLoader(valset, batch_size=64, shuffle=False)


###################################################################################
###################################################################################
train_dataset_length = len(fashion_trainloader.dataset)
val_dataset_length = len(fashion_valloader.dataset)
print("Fashion dataset")
print("Length of training dataset:", train_dataset_length)
print("Length of validation dataset:", val_dataset_length)

train_dataset_length = len(mnist_trainloader.dataset)
val_dataset_length = len(mnist_valloader.dataset)

# Collect labels from both loaders
train_labels = get_all_labels(fashion_trainloader)
val_labels = get_all_labels(fashion_valloader)

# Now you can print or inspect the labels
print("Unique labels in train loader:", sorted(set(train_labels)))
print("Unique labels in validation loader:", sorted(set(val_labels)))

print("Mnist dataset")
print("Length of training dataset:", train_dataset_length)
print("Length of validation dataset:", val_dataset_length)

train_labels = get_all_labels(mnist_trainloader)
val_labels = get_all_labels(mnist_valloader)

# Now you can print or inspect the labels
print("Unique labels in train loader:", sorted(set(train_labels)))
print("Unique labels in validation loader:", sorted(set(val_labels)))
# Model initialization

#model = KAN([28 * 28, 64, 20]).to(device)
model = MLP(classes=20).to(device)

# Optimizer and loss function
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

criterion = nn.CrossEntropyLoss()


# Training loop
def train_task(task_num, epochs=10, train_loader = mnist_trainloader, test_loader = mnist_valloader):
    
    for epoch in range(epochs):  # Train each task for 5 epochs
        model.train()
        train_loss, train_accuracy = 0, 0
        for images, labels in tqdm(train_loader):
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
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




# Execute training for both tasks

print("Fashion dataset")
train_task(1, epochs=5, train_loader = fashion_trainloader, test_loader = fashion_valloader)
for param in model.parameters():
    param.requires_grad = False

for param in model.layers[-1].parameters():
    param.requires_grad = True
optimizer = optim.AdamW(model.parameters(), lr=6e-5, weight_decay=2e-6)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
print("Mnist dataset")
train_task(2, epochs=5, train_loader = mnist_trainloader, test_loader = mnist_valloader)


model.eval()
total_accuracy = 0

with torch.no_grad():
    for images, labels in tqdm(fashion_valloader):
        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'Overall Test Accuracy on Fashion: {total_accuracy / len(fashion_valloader):.4f}')

total_accuracy = 0
with torch.no_grad():
    for images, labels in tqdm(mnist_valloader):
        images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
        outputs = model(images)
        total_accuracy += (outputs.argmax(1) == labels).float().mean().item()
print(f'Overall Test Accuracy on MNIST: {total_accuracy / len(mnist_valloader):.4f}')
