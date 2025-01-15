# Ovarian-canser-classification

A deep learning-based project aimed at classifying ovarian cancer subtypes using advanced image processing techniques, convolutional neural networks (CNN), and TensorRT optimization.

## Overview

Ovarian cancer is one of the deadliest gynecological cancers, often diagnosed in later stages due to the lack of early symptoms and limitations in diagnostic technologies. This project focuses on identifying five subtypes of ovarian cancer using AI models, aiding in early and accurate diagnosis. The subtypes include:

- **High-Grade Serous Carcinoma**
- **Clear Cell Ovarian Carcinoma**
- **Endometrioid Carcinoma**
- **Low-Grade Serous Carcinoma**
- **Mucinous Carcinoma**

This study utilizes convolutional neural networks (CNN) and architectures such as ResNet50, VGG16, and VGG19 to classify the cancer subtypes effectively. Additionally, TensorRT optimization has been implemented to enhance model inference speed, making it suitable for real-time applications.

## Features

- **Data Preprocessing**: Includes image resizing, normalization, augmentation (rotation, flipping, zooming), noise reduction, contrast, and brightness adjustments.
- **Deep Learning Models**: Implements custom CNN architectures, ResNet50, VGG16, and VGG19 for classification.
- **Model Optimization**: Utilizes techniques like L1 regularization, class balancing, dropout, early stopping, and learning rate scheduling for efficient training.
- **TensorRT Optimization**: Optimizes the ResNet50 model using Torch-TensorRT with FP16 precision for faster inference.
- **Performance Metrics**: Evaluates models using accuracy, precision, recall, and F1-score.
- **Visualization**: Displays training/validation accuracy and loss for performance analysis.

## Dataset

The dataset contains over 2,900 pathological images for each cancer subtype, sourced from the Kaggle platform. Preprocessing steps include:

- **Image resizing**: All images are resized to 224x224 pixels.
- **Normalization**: Pixel values are normalized to the range [0, 1].
- **Augmentation**: Techniques like rotation, flipping, and zooming are applied to reduce overfitting.

## Models and Techniques

Several models and techniques are implemented in this project:

1. **Custom CNN**:  
   - Multiple convolutional layers with 3x3 filters, max-pooling, dropout, and batch normalization layers to prevent overfitting.
   - Fully connected layers for classification.
   
2. **Pretrained Architectures**:  
   - **ResNet50**: Fine-tuned for ovarian cancer subtype classification, achieving a high accuracy of **96%** with Keras and TensorFlow.  
   - **VGG16 and VGG19**: Customized versions of VGG models with additional dropout and batch normalization layers.

3. **TensorRT Optimization**:  
   - Optimized the ResNet50 model with Torch-TensorRT for FP16 precision, significantly reducing inference time for real-time applications.

### Model Evaluation

- **ResNet50**: Achieved the highest accuracy (**96%**) and fastest training time. Optimized with TensorRT for enhanced inference speed.
- **VGG16 and VGG19**: Achieved moderate accuracies but required longer training times.
- **Custom CNN**: Provided flexibility with decent performance but did not outperform ResNet50.

### Techniques Used

- L1 regularization for weight penalties.
- Class balancing to handle imbalanced datasets.
- Early stopping to prevent overfitting.
- Learning rate scheduling to improve training efficiency.

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

4. **Optimizing with TensorRT**:
   - Model is optimized with different batch_sizes.
   - Inference times were compared with different batch sizes and with both non-optimized and optimized models.

5. **Comparison**:
   - ResNet50 outperformed other architectures in accuracy, training time, and consistency.

## Results

- ResNet50 achieved the highest accuracy and stability, making it the most suitable architecture for this classification task.
- VGG16 and VGG19 showed lower performance and higher computational costs.
