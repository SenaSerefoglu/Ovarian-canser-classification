from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Reshape
from keras.regularizers import l2, l1
from keras.applications import ResNet50, VGG19, VGG16

def CNNmodel():
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(128, (3, 3), input_shape=[224, 224, 3], activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(32,(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(16,(3,3), activation='relu', kernel_regularizer=l2(0.01)))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())

    # Fully connected layer with L1 regularization and Dropout
    model.add(Dense(64, activation='relu', kernel_regularizer=l1(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))
    
    return model

def ResNet50_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


def ResNet50_custom_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add dense layers directly after global average pooling
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Fully connected layer with L1 regularization and Dropout
    x = Dense(64, activation='relu', kernel_regularizer=l1(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def VGG19_custom_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add dense layers directly after global average pooling
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Fully connected layer with L1 regularization and Dropout
    x = Dense(64, activation='relu', kernel_regularizer=l1(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def VGG16_custom_model(input_shape=(224, 224, 3), num_classes=5):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add dense layers directly after global average pooling
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(16, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Fully connected layer with L1 regularization and Dropout
    x = Dense(64, activation='relu', kernel_regularizer=l1(0.01))(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model