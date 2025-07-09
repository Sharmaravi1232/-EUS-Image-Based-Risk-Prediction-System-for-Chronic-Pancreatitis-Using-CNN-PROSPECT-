import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image

# Configure computation device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to CSV containing flattened MNIST images
CSV_PATH = "/content/sample_data/mnist_train_small.csv"

class CSVToImageDataset(Dataset):
    """
    Custom Dataset class to load image and label data from a CSV file.
    The first column is the label, remaining columns are pixel values.
    """
    def __init__(self, dataframe, transform=None):
        self.labels = dataframe.iloc[:, 0].values
        self.images = dataframe.iloc[:, 1:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx].reshape(28, 28).astype(np.uint8)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label

# Preprocessing pipeline to adapt grayscale 28x28 images to 3-channel 299x299
preprocessing = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset and split into training and validation sets
df = pd.read_csv(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df.iloc[:, 0], random_state=42)

train_dataset = CSVToImageDataset(train_df, transform=preprocessing)
val_dataset = CSVToImageDataset(val_df, transform=preprocessing)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load InceptionV3 with pretrained weights
model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 10)
model.aux_logits = True
model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, 10)
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def evaluate_model(model, data_loader):
    """
    Evaluates model on a given DataLoader.
    Returns accuracy and weighted F1-score.
    """
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

# Training loop
num_epochs = 5
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, aux_outputs = model(images)
        loss_main = criterion(outputs, labels)
        loss_aux = criterion(aux_outputs, labels)
        total_loss = loss_main + 0.4 * loss_aux
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

    val_accuracy, val_f1 = evaluate_model(model, val_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {running_loss:.4f} | "
          f"Validation Accuracy: {val_accuracy:.4f} | Validation F1 Score: {val_f1:.4f}")

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), "best_inception_mnist.pth")

print(f"Training completed. Best Validation Accuracy: {best_accuracy:.4f}")
