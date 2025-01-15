from utils import (
    load_and_preprocess_images,
    calculate_class_weights,
    create_dataloaders,
    visualize_results
)
from utils import MODEL
from models import VGG16CustomModel, VGG19CustomModel, ResNet50CustomModel, CNNmodel
from TRT import TensorRTModelOptimizer
import numpy as np
import torch
import time
import os


def train_save():
    # Define the directories for training and testing data
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_dir = os.path.join(current_dir, '..', 'Test_Images')
    train_dir = os.path.join(current_dir, '..', 'Train_Images')

    print("Loading and preprocessing images...")

    # Load and preprocess the images
    train_images, train_labels = load_and_preprocess_images(train_dir)
    print(f"Train images loaded: {len(train_images)} images")

    test_images, test_labels = load_and_preprocess_images(test_dir)
    print(f"Test images loaded: {len(test_images)} images")

    # Calculate class weights for imbalanced datasets
    class_weights = calculate_class_weights(train_labels)
    print("Class weights calculated")

    # Create data loaders
    train_loader, test_loader = create_dataloaders(train_images, train_labels, test_images, test_labels)
    print("Data loaders created")

    # Choose the model
    model_name = "ResNet50"  # Change to "CNNmodel", "VGG16", or "VGG19" for other models
    if model_name == "CNNmodel":
        model = CNNmodel(num_classes=len(np.unique(train_labels)))
    elif model_name == "ResNet50":
        model = ResNet50CustomModel(num_classes=len(np.unique(train_labels)))
    elif model_name == "VGG16":
        model = VGG16CustomModel(num_classes=len(np.unique(train_labels)))
    elif model_name == "VGG19":
        model = VGG19CustomModel(num_classes=len(np.unique(train_labels)))
    else:
        raise ValueError("Invalid model name")
    print(f"{model_name} model created")

    # Wrap the model with the MODEL utility class
    model = MODEL(model, train_loader, test_loader, class_weights=class_weights)

    # Train the model
    print("Starting model training...")
    history = model.train_model(lambda_l1=0.01, epochs=50, patience=10, monitor='acc')
    print("Model training completed")

    # Evaluate the model
    test_loss, test_acc = model.evaluate_model()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    model.save(file_name=f"{model_name}_model.pt")
    model.save_weights(file_name=f"{model_name}_model_weights.pt")
    model.save_script(file_name=f"{model_name}_model_script.pt")
    print(f"{model_name} model saved.")

    # Visualize training history
    visualize_results(history)


def TorchRT():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, '..', 'models\\ResNet50_model_script.pt')

    optimizer = TensorRTModelOptimizer(model_dir)
    optimizer.optimize_model()
    optimizer.save_optimized_model(save_dir='ResNet50_TensorRT.pt')

def inference():
    # Load the optimized model
    optimized_model = torch.jit.load('ResNet50_TensorRT.pt')
    optimized_model.eval()
    optimized_model.cuda()


    # Inference for test data
    batch_sizes = [1, 4, 8, 16]
    for batch in batch_sizes:
        start_time = time.time()
        input_data = np.random.rand(batch, 3, 224, 224).astype(np.float32)
        input_tensor = torch.from_numpy(input_data).cuda().half()  # Ensure input tensor matches FP16 precision

        # Perform inference
        with torch.no_grad():
            output = optimized_model(input_tensor)

        print(f"Batch size: {batch}, Çıktı boyutu: {output.size()}")

        # Calculate inference time
        inference_time = time.time() - start_time

        # Normalize inference time
        inference_time /= batch

        print(f"Batch size: {batch}, Inference Time: {inference_time:.4f} s")
    
    resnet_model = torch.load('models/ResNet50_model.pt')
    resnet_model.eval()
    resnet_model.cuda()

    # Inference for test data  
    for batch in batch_sizes:
        start_time = time.time()
        input_data = np.random.rand(batch, 3, 224, 224).astype(np.float32)
        input_tensor = torch.from_numpy(input_data).cuda()

        # Perform inference
        with torch.no_grad():
            output = resnet_model(input_tensor)

        print(f"Batch size: {batch}, Output size: {output.size()}")

        # Calculate inference time
        inference_time = time.time() - start_time

        # Normalize inference time
        inference_time /= batch

        print(f"Batch size: {batch}, Inference Time: {inference_time:.4f} s")


if __name__ == '__main__':
    train_save()
