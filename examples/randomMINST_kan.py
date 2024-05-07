import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from efficient_kan import KAN  # Make sure this import is correct
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
    
class MNISTDataset(Dataset):
    def __init__(self, mnist_dataset, is_train=True):
        self.dataset = mnist_dataset
        self.is_train = is_train
        self.permutation = None

    def apply_permutation(self):
        if self.permutation is None and self.is_train:
            # Generate permutation only for training dataset
            self.permutation = torch.randperm(len(self.dataset.targets))
        if self.is_train:
            # Apply permutation to only the training dataset
            self.dataset.targets = self.dataset.targets[self.permutation]

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
    
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

# Load MNIST datasets
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainset = MNISTDataset(train_data, is_train=True)
valset = MNISTDataset(val_data, is_train=False)

trainloader = DataLoader(trainset, batch_size=64, shuffle=False)
valloader = DataLoader(valset, batch_size=64, shuffle=False)

#model = KAN([28 * 28, 16, 10])
model = MLP(classes=10)

total_params, trainable_params = count_parameters(model)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(20):
    if epoch == 10:
        trainset.apply_permutation()  # Apply permutation at 10th epoch
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    with tqdm(trainloader, desc=f"Epoch {epoch + 1}") as pbar:
        for images, labels in pbar:
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            total_loss += loss.item()
            accuracy = 100. * correct / total
            pbar.set_postfix(loss=total_loss / total, accuracy=accuracy)

    scheduler.step()

    # Optionally, you can also print validation accuracy here, if needed
