from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, TensorBoard
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import numpy as np

def compile_model(model):
    optim = Adam(learning_rate=0.0001)
    model.compile(optimizer=optim, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def calculate_class_weights(labels):
    classes = np.unique(labels)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=labels)
    class_weights_dict = dict(zip(classes, class_weights))
    return class_weights_dict

def train_model(model, train_generator, validation_generator, class_weights):
    #Implement lr scheduler
    def lr_scheduler(epoch, lr):
        if epoch < 10:
            return lr
        else:
            return float(lr * tf.math.exp(-0.1))
        
    
    lr_schedule = LearningRateScheduler(lr_scheduler, verbose=1)
    
    # Implement early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )

    # Implement model checkpoint
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    # Monitor training
    monitor = TensorBoard(log_dir='logs')

    # Train the model
    history = model.fit(
        train_generator,
        batch_size=32,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[monitor, lr_schedule, early_stopping, model_checkpoint],
        class_weight = class_weights
    )

    return history


def evaluate_model(model, test_images, test_labels):
    # Evaluate the model
    loss, accuracy = model.evaluate(test_images, test_labels)
    return loss, accuracy


def visualize_results(history):
    # Plot the training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot the training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    

def save_model(model, file_name='final_model.keras'):
    model.save(file_name)
    print('Model saved')

def create_generators(train_images, train_labels, test_images, test_labels):
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    validation_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

    return train_generator, validation_generator



def preprocess_image(image):
    blured_image = cv2.fastNlMeansDenoising(image, None, 6, 6, 7)

    # Kontrast Ayarlama (CLAHE)
    lab = cv2.cvtColor(blured_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Parlaklık ve Kontrast Ayarı
    alpha = 0.76  # Kontrast kontrolü
    beta = 15    # Parlaklık kontrolü
    adjusted = cv2.convertScaleAbs(image_clahe, alpha=alpha, beta=beta)

    return adjusted

def load_and_preprocess_images(directory):
    images = []
    labels = []

    classes = os.listdir(directory)
    for idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory,class_name)
        image_count = 0
        for image_name in os.listdir(class_dir):
            if image_count > 2900:
                break
            image = cv2.imread(os.path.join(directory,class_name,image_name))
            image = cv2.resize(image, (224, 224))   # Ensure the images are resized to 224x224
            image = preprocess_image(image)
            images.append(image)
            labels.append(idx)
            image_count += 1
    images = np.array(images)/255
    labels = np.array(labels)

    return images, labels