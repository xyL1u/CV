import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from model import *
import pickle
from PIL import Image
from pathlib import Path

class CustomCIFAR100Dataset(data.Dataset):
    def __init__(self, data, label, transform=None, train=True):
        super().__init__()
        self.data = data
        self.label = label
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img.astype('uint8'))

        if self.transform:
            img = self.transform(img)

        if self.train:
            label = self.label[idx]
            return img, label
        else:
            return img



# Data preparation
transform_train = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

transform_test = transforms.Compose([transforms.Resize((32, 32)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_file = 'D:\\LearnPytorch\\LearnPytorch\\ImageClassification\\CIFAR100\\train'
test_file = 'D:\\LearnPytorch\\LearnPytorch\\ImageClassification\\CIFAR100\\test'
label_file = 'D:\\LearnPytorch\\LearnPytorch\\ImageClassification\\CIFAR100\\meta'

# Load train and test dictionaries
train_dict = pickle.load(Path(train_file).open('rb'), encoding='bytes')
test_dict = pickle.load(Path(test_file).open('rb'), encoding='bytes')
labels = pickle.load(Path(label_file).open('rb'))

# Extract and process train/test data
train_data = train_dict[b'data']  # Numpy array of shape (50000, 3072)
train_labels = train_dict[b'fine_labels']  # Fine-grained labels (100 classes)
test_data = test_dict[b'data']  # Numpy array of shape (10000, 3072)

full_train_dataset = CustomCIFAR100Dataset(train_data, train_labels, transform=transform_train, train=True)
test_dataset = CustomCIFAR100Dataset(test_data, label=None, transform=transform_test, train=False)

val_size = int(0.2 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = data.random_split(full_train_dataset, [train_size, val_size])


batch_size = 64
train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Hyperparameters
lr = 1e-4
num_epochs = 10

# Model, loss function and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LeNet(num_classes=100).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f'Traning'):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Track statistics
            running_loss += loss.item() * inputs.size(0)
            _,predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_accuracy = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}')

        # Validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_running_loss / val_total
        val_epoch_accuracy = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        print(f'Epoch {epoch + 1}/{num_epochs} - Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_epoch_accuracy:.4f}')

        # Save best model
        if val_epoch_accuracy > best_acc:
            best_acc = val_epoch_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_epoch_loss,
                'accuracy': val_epoch_accuracy
            }, 'best_model.pth')
            print(f'Saved new best model with accuracy: {val_epoch_accuracy:.4f}')
        print('-' * 60)

def inference(model, test_loader):
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for inputs, _ in tqdm(test_loader, desc='Testing'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())

    return all_predictions

def plot_training():
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12,5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss', color='g', linestyle='-')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', linestyle='-')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy', color='g', linestyle='-')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='orange', linestyle='-')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

train_and_validate(model, train_loader, val_loader, criterion, optimizer, num_epochs)
plot_training()

# Test the model and save predictions
best_model = LeNet(num_classes=10).to(device)
best_model.load_state_dict(torch.load('best_model.pth'))
test_predictions = inference(best_model, test_loader)

df = pd.DataFrame({'id': range(len(test_predictions)),
                   'label': test_predictions})
df.to_csv('test_predictions.csv', index=False)
print('Predictions are successfully saved')