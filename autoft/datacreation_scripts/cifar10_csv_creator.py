import os

import src.templates as templates
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Using the torchvision datasets to download or load CIFAR10
root = '/iris/u/cchoi1/Data'
image_save_path = '/iris/u/cchoi1/Data/CIFAR-10'
if not os.path.exists(root):
    os.makedirs(root)
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

transform = transforms.Compose([transforms.ToTensor()])  # Basic transform to get tensor data
dataset = CIFAR10(root=root, train=True, transform=transform, download=True)

# Define the CSV output file and write the header
csv_path = "/iris/u/cchoi1/Data/csv/cifar10.csv"
out = open(csv_path, "w")
out.write("title\tfilepath\tlabel\n")

template = getattr(templates, 'simple_template')

# Loop through the dataset to save the images and write to CSV
for i, (image, label) in enumerate(dataset):
    class_name = classes[label]

    # Define the image path and save the tensor as an image
    image_folder_path = os.path.join(image_save_path, class_name)
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)

    # Convert tensor back to PIL Image and save
    image_pil = transforms.ToPILImage()(image)
    image_file_path = os.path.join(image_folder_path, f'image_{i}.png')
    image_pil.save(image_file_path)

    for t in template:
        caption = t(class_name)
        out.write("%s\t%s\t%s\n" % (caption, image_file_path, label))

out.close()


