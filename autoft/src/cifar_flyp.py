import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from CLIP import clip

# Load CLIP model
accumulation_steps = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# CIFAR-10 class labels with descriptive text templates
cifar10_labels = ["a photo of an airplane", "a photo of an automobile", "a photo of a bird",
                  "a photo of a cat", "a photo of a deer", "a photo of a dog",
                  "a photo of a frog", "a photo of a horse", "a photo of a ship",
                  "a photo of a truck"]


# Custom dataset class for CIFAR-10 with text captions
class CIFAR10Dataset(Dataset):
    def __init__(self, cifar_dataset):
        self.cifar_dataset = cifar_dataset

    def __len__(self):
        return len(self.cifar_dataset)

    def __getitem__(self, idx):
        image, label = self.cifar_dataset[idx]
        text = cifar10_labels[label]
        return image, text


# Load and preprocess CIFAR-10 dataset
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataset = CIFAR10Dataset(trainset)
trainloader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the contrastive loss
def contrastive_loss(image_features, text_features):
    logits = (image_features @ text_features.T) / model.logit_scale.exp()
    labels = torch.arange(len(logits), device=device)
    loss_img = torch.nn.functional.cross_entropy(logits, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits.T, labels)
    return (loss_img + loss_txt) / 2


# Training loop
optimizer = optim.Adam(model.parameters(), lr=3e-5)

for epoch in range(10):
    model.train()
    for images, texts in trainloader:
        images = images.to(device)
        texts = clip.tokenize(texts).to(device)

        optimizer.zero_grad()
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        loss = contrastive_loss(image_features, text_features)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1} completed")

# Evaluation function and code (similar to previous scripts)
# ...

# Evaluate on CIFAR-10.1 and CIFAR-10.2
# ...

# Save the model
model.save("cifar_flyp_vitl14.pt")