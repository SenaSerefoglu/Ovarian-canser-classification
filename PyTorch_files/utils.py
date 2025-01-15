import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from sklearn.utils.class_weight import compute_class_weight

# Custom Dataset
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Calculate class weights
def calculate_class_weights(labels):
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    return class_weights_tensor

def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

class MODEL():
    def __init__(self, model, train_loader, test_loader, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        if class_weights is not None:
            self.class_weights = class_weights 
    
    # Compile model
    def compile_model(self, learning_rate=0.0001):
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()
        return optimizer, criterion

    # Train model
    def train_model(self, lambda_l1=0.0, epochs=50, patience=5, monitor='acc'):
        optimizer, criterion = self.compile_model()
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Assign class weights to the criterion
        if self.class_weights is not None:
            class_weights = self.class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)

        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        # Initialize history for tracking loss and accuracy
        best_val_metric = -float('inf') if monitor == 'acc' else float('inf')  # Initialize the best metric for accuracy or loss
        patience_counter = 0  # To track the number of epochs without improvement

        for epoch in range(epochs):
            # Training Phase
            self.model.train()
            train_loss, train_correct = 0.0, 0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = self.model(images)
                loss = criterion(outputs, labels)
                if lambda_l1 > 0:
                    loss += l1_regularization(self.model, lambda_l1)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()

            train_loss /= len(self.train_loader.dataset)
            train_acc = train_correct / len(self.train_loader.dataset)

            # Validation Phase
            self.model.eval()
            val_loss, val_correct = 0.0, 0
            with torch.no_grad():
                for images, labels in self.test_loader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()

            val_loss /= len(self.test_loader.dataset)
            val_acc = val_correct / len(self.test_loader.dataset)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")

            # Early Stopping Logic Based on Accuracy or Loss
            if monitor == 'acc':
                metric = val_acc  # Monitor validation accuracy
                if metric > best_val_metric:
                    best_val_metric = metric
                    patience_counter = 0  # Reset patience counter if improvement is seen
                else:
                    patience_counter += 1
            else:
                metric = val_loss  # Monitor validation loss
                if metric < best_val_metric:
                    best_val_metric = metric
                    patience_counter = 0  # Reset patience counter if improvement is seen
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping after {epoch + 1} epochs due to no improvement in validation {monitor}.")
                break

            # Learning Rate Scheduler Step
            scheduler.step()

        return history

    # Evaluate model
    def evaluate_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        test_loss, test_correct = 0.0, 0
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = self.model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()

        test_loss /= len(self.test_loader.dataset)
        test_acc = test_correct / len(self.test_loader.dataset)

        return test_loss, test_acc
    
    # Save model weights
    def save_weights(self, file_name='final_model_weights.pt'):
        torch.save(self.model.state_dict(), file_name)
        print('Model weights saved')

    def save(self, file_name='final_model.pt'):
        torch.save(self.model, file_name)
        print('Model saved')

    def save_script(self, file_name='final_model_script.pt'):
        model_script = torch.jit.script(self.model)
        model_script.save(file_name)
        print('Model script saved')

# Visualize results
def visualize_results(history):
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Preprocess and load images
def preprocess_image(image):
    blured_image = cv2.fastNlMeansDenoising(image, None, 6, 6, 7)

    # Contrast Adjustment (CLAHE)
    lab = cv2.cvtColor(blured_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Brightness and Contrast Adjustment
    alpha = 0.76  # Contrast control
    beta = 15    # Brightness control
    adjusted = cv2.convertScaleAbs(image_clahe, alpha=alpha, beta=beta)

    return adjusted

def load_and_preprocess_images(directory):
    images, labels = [], []

    classes = os.listdir(directory)
    for idx, class_name in enumerate(classes):
        print("Loading class: ", class_name)
        class_dir = os.path.join(directory, class_name)
        image_count = 0
        for image_name in os.listdir(class_dir):
            if image_count % 100 == 0 and image_count > 0:
                print("Loading image: ", image_count)
            if image_count > 12000:
                break
            image = cv2.imread(os.path.join(class_dir, image_name))
            image = cv2.resize(image, (224, 224))  # Ensure the images are resized to 224x224
            image = preprocess_image(image)
            images.append(image)
            labels.append(idx)
            image_count += 1
    images = np.array(images, dtype=np.float32) / 255
    labels = np.array(labels)

    return images, labels

# Create data loaders
def create_dataloaders(train_images, train_labels, test_images, test_labels):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = CustomImageDataset(train_images, train_labels, transform=transform)
    test_dataset = CustomImageDataset(test_images, test_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader
