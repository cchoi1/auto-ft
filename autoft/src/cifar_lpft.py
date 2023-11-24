import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from CLIP import clip
from src.datasets.cifar10 import CIFAR101, CIFAR102

accumulation_steps = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
model, train_preprocess, val_preprocess = clip.load("ViT-L/14", device=device, jit=False)

trainset = torchvision.datasets.CIFAR10(root='/iris/u/cchoi1/Data', train=True, download=True, transform=train_preprocess)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
cifar10_1_set = CIFAR101(val_preprocess, train=False, n_examples=-1, location="/iris/u/cchoi1/Data").dataset
cifar10_1_loader = DataLoader(cifar10_1_set, batch_size=64, shuffle=False)
cifar10_2_set = CIFAR102(val_preprocess, train=False, n_examples=-1, location="/iris/u/cchoi1/Data").dataset
cifar10_2_loader = DataLoader(cifar10_2_set, batch_size=64, shuffle=False)

model.eval()  # Start with model in evaluation mode

# Linear Probe Training: Only train the classification head
for param in model.parameters():
    param.requires_grad = False

# Add and train a new classification head (assuming CIFAR-10 has 10 classes)
num_features = model.visual.output_dim  # Get the number of features from CLIP model
model.fc = torch.nn.Linear(num_features, 10).to(device)
optimizer = optim.Adam(model.fc.parameters(), lr=3e-5)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000*accumulation_steps, eta_min=0)
def train_one_epoch(epoch_index):
    model.train()  # Set the model to training mode
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


# Full Fine-Tuning: Unfreeze the entire model
for param in model.parameters():
    param.requires_grad = True
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training and Evaluation Loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    accuracy_cifar10_1 = evaluate_model(model, cifar10_1_loader)
    accuracy_cifar10_2 = evaluate_model(model, cifar10_2_loader)
    print(f'Epoch {epoch+1}, CIFAR-10.1 Accuracy: {accuracy_cifar10_1}%, CIFAR-10.2 Accuracy: {accuracy_cifar10_2}%')

# Save the model
model.save("cifar_lpft_vitl14.pt")