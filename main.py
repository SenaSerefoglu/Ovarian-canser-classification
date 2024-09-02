from utils import *
from models import VGG16_custom_model, VGG19_custom_model,ResNet50_custom_model, CNNmodel, ResNet50

def main():
    # Load the data
    train_dir = 'Train_Images'
    test_dir = 'Test_Images'

    # Load and preprocess images
    train_images, train_labels = load_and_preprocess_images(train_dir)
    test_images, test_labels = load_and_preprocess_images(test_dir)

    class_weights = calculate_class_weights(train_labels)

    train_generator, validation_generator = create_generators(train_images, train_labels, test_images, test_labels)

    model = VGG19_custom_model()
    compile_model(model)
    history = train_model(model, train_generator, validation_generator, class_weights)
    evaluate_model(model, test_images, test_labels)
    visualize_results(history)
    save_model(model, "yedek/resnet_model.keras")


if __name__ == '__main__':
    main()