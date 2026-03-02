import torch, torchvision
import torchvision.transforms as transforms
import tqdm
from PIL import Image
import os

from transformers import ViTForImageClassification
from transformersurgeon import ViTForImageClassificationCompress, ViTCompressionSchemesManager

DO_COMPRESSION = True
VERBOSE = True
USE_GPU = True
BATCH_SIZE = 512

# Device configuration
device = torch.device("cuda" if (torch.cuda.is_available() and USE_GPU) else "cpu")

# Set dataset transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# Load imagenet dataset

file_name = os.path.join(os.path.dirname(__file__), "imagenet_location.txt")
with open(file_name, 'r') as f:
    imagenet_path = f.readline().strip()
imagenet_dataset = torchvision.datasets.ImageNet(imagenet_path, split='val', transform=transform)

# Prepare dataset loader
data_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# Load pre-trained compressed ViT model
model = ViTForImageClassificationCompress.from_pretrained('google/vit-base-patch16-224')

if DO_COMPRESSION:
    # Instantiate compression manager
    manager = ViTCompressionSchemesManager(model)
    
    # Apply compression schemes
    manager.set_pruning_ratio_all(0.9, verbose=VERBOSE)
    manager.set_pruning_mode_all("unstructured", verbose=VERBOSE)
    
    # Apply all compression schemes to the model
    manager.apply_all(hard=False, verbose=VERBOSE)

    # Update in-place the compressed model configuration from the manager
    manager.update_config()

# Perform evaluation
model.eval()
model.to(device)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in tqdm.tqdm(data_loader, "Evaluating accuracy..."):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images).logits
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Accuracy of the model on the ImageNet validation set: {accuracy:.2f}%')

