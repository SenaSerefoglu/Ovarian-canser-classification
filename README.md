# Ovarian-canser-classification

A deep learning-based project aimed at classifying ovarian cancer subtypes using advanced image processing techniques and convolutional neural networks (CNN).

## Overview

Ovarian cancer is one of the deadliest gynecological cancers, often diagnosed in later stages due to the lack of early symptoms and limitations in diagnostic technologies. This project focuses on identifying five subtypes of ovarian cancer using AI models, aiding in early and accurate diagnosis. The subtypes include:

- **High-Grade Serous Carcinoma**
- **Clear Cell Ovarian Carcinoma**
- **Endometrioid Carcinoma**
- **Low-Grade Serous Carcinoma**
- **Mucinous Carcinoma**

This study utilizes convolutional neural networks (CNN) and architectures such as ResNet50, VGG16, and VGG19 to classify the cancer subtypes effectively.

## Features

- **Data Preprocessing**: Includes image resizing, normalization, and augmentation techniques (e.g., rotation, flipping, zooming).
- **Deep Learning Models**: Implements CNN-based architectures (e.g., ResNet50, VGG16, VGG19) for classification.
- **Model Optimization**: Utilizes techniques like class weights, dropout, early stopping, and learning rate scheduling.
- **Performance Metrics**: Evaluates models using accuracy, precision, recall, and F1-score.
- **Visualization**: Displays training and validation metrics for performance analysis.

## Dataset

The dataset contains over 2,900 pathological images for each cancer subtype, sourced from the Kaggle platform. Preprocessing steps include:

- **Image resizing**: All images are resized to 224x224 pixels.
- **Normalization**: Pixel values are normalized to the range [0, 1].
- **Augmentation**: Techniques like rotation, flipping, and zooming are applied to reduce overfitting.

## Models

Several models were implemented and compared:

1. **Custom CNN**: A custom convolutional neural network with multiple convolutional, pooling, and fully connected layers.
2. **ResNet50**: Achieved the best performance with higher accuracy and faster training time.
3. **VGG16**: High training time with less satisfactory results.
4. **VGG19**: Stable learning but lower accuracy compared to ResNet50.

### Model Architecture Highlights

- **Custom CNN**:
  - Multiple convolutional layers with 3x3 filters.
  - Max pooling, dropout, and batch normalization layers to prevent overfitting.
  - Fully connected layers for classification.

- **Pretrained Architectures**:
  - Used ResNet50, VGG16, and VGG19 pretrained on ImageNet for feature extraction.
  - Fine-tuned for the ovarian cancer subtype classification task.

### Model Evaluation

- **Best Performing Model**: ResNet50
  - High accuracy with stable and consistent results.
  - Faster training time compared to VGG architectures.
- **VGG16 and VGG19**:
  - Longer training times and lower accuracy.
  - Less efficient compared to ResNet50.

## Methodology

1. **Data Collection and Preprocessing**:
   - Images were sourced from Kaggle and processed for model input.
   - Techniques such as noise reduction, contrast adjustment, and brightness control were applied.

2. **Model Training**:
   - Models were trained using TensorFlow and Keras.
   - Optimizer: Adam (learning rate = 0.0001).
   - Loss function: Sparse categorical cross-entropy.
   - Techniques like early stopping, learning rate scheduling, and model checkpointing were used.

3. **Performance Evaluation**:
   - Metrics: Accuracy, precision, recall, F1-score.
   - Training and validation results were visualized for analysis.

4. **Comparison**:
   - ResNet50 outperformed other architectures in accuracy, training time, and consistency.

## Results

- ResNet50 achieved the highest accuracy and stability, making it the most suitable architecture for this classification task.
- VGG16 and VGG19 showed lower performance and higher computational costs.
