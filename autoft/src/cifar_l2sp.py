import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import clip.clip as clip
from src.datasets.cifar10 import CIFAR101, CIFAR102
from src.models.modeling import ImageClassifier

accumulation_steps = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
model, train_preprocess, val_preprocess = ImageClassifier.load("/iris/u/cchoi1/robust-optimizer/autoft/zeroshot/clip_vitb16_cifar10.pt")

trainset = torchvision.datasets.CIFAR10(root='/iris/u/cchoi1/Data', train=True, download=True, transform=train_preprocess)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

cifar10_1_set = CIFAR101(val_preprocess, train=False, n_examples=-1, location="/iris/u/cchoi1/Data").dataset
cifar10_1_loader = DataLoader(cifar10_1_set, batch_size=64, shuffle=False)

cifar10_2_set = CIFAR102(val_preprocess, train=False, n_examples=-1, location="/iris/u/cchoi1/Data").dataset
cifar10_2_loader = DataLoader(cifar10_2_set, batch_size=64, shuffle=False)

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model.encode_image(images)
            # Assuming the model outputs logits or probabilities
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Define L2-SP Regularization
def l2_sp_regularizer(model, original_model, strength=1e-1):
    loss = 0.0
    for (name, param), (_, original_param) in zip(model.named_parameters(), original_model.named_parameters()):
        loss += torch.sum((param - original_param) ** 2)
    return strength * loss

# Clone the original model for L2-SP regularization
original_model = clip.load("ViT-L/14", device=device, jit=False)[0]
original_model.load_state_dict(model.state_dict())

# Training
optimizer = optim.Adam(model.parameters(), lr=3e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000*accumulation_steps, eta_min=0)
for epoch in range(10):
    for i, (images, labels) in enumerate(trainloader):
        images = images.to(device)
        optimizer.zero_grad()
        logits = model.encode_image(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss += l2_sp_regularizer(model, original_model)
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(trainloader):
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
    print(f"Epoch {epoch+1} completed")

    accuracy_cifar10_1 = evaluate_model(model, cifar10_1_loader)
    print(f'Accuracy on CIFAR-10.1: {accuracy_cifar10_1}%')
    accuracy_cifar10_2 = evaluate_model(model, cifar10_2_loader)
    print(f'Accuracy on CIFAR-10.2: {accuracy_cifar10_2}%')

# Save the model
torch.save(model.state_dict(), "cifar_l2sp_vitl14.pt")