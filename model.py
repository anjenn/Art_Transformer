import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import layers

import utils

CONTENT_IMG = 'content.png'  # Update path
STYLE_IMG = 'style.png'  # Update path
CONTENT_DIR = "./content"
STYLE_DIR = "./style"

print("Checking Content Directory:", os.listdir(CONTENT_DIR))
print("Checking Style Directory:", os.listdir(STYLE_DIR))

def train_step(content_image, style_image, generated_image, content_weight, style_weight):
    optimizer = tf.optimizers.Adam(learning_rate=0.02) # !Should be reinitialised at every step to handle new var when dealing with many data

    with tf.GradientTape() as tape:
        # Extract features for content and style images
        content_features, style_features = extract_features(content_image, style_image)
        # generated_features = model(generated_image) # For a single image
        generated_features = extract_features(generated_image, generated_image)

        print('DEBUGGING!!!!!!!!!!!!!!!!!!')
        print(type(style_features))  
        print(type(content_features))  
        # style_features = tf.convert_to_tensor(style_features)
        # style_gram = utils.gram_matrix(style_features)
        # style_features = tf.reduce_mean(style_features)

        # Cost = content + style loss
        # loss = utils.compute_loss(content_weight, style_weight, content_features, style_features, generated_features) # For a single image

        loss = utils.compute_loss(content_features, style_features, generated_features, content_weight, style_weight)

    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])

    return loss

# Build the CNN model from scratch
def build_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential()

    # For CNNs or complex models: Use model.add()
    
    # First few layers (Convolutional layers)
    model.add(layers.InputLayer(input_shape=input_shape))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Deeper layers
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Additional convolutional blocks as needed
    model.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
    
    return model

# Create the model
model = build_model()
# Adam optimizer to optimize the generated image
# optimizer = tf.optimizers.Adam(learning_rate=0.02)

# Function to extract features from the model
def extract_features(content_image, style_image):
    content_features = model(content_image)
    style_features = model(style_image)
    
    return content_features, style_features


def main():
    # ########################################
    # # # Single Training Image case
    # ########################################
    # # Initial Image Display
    # content_image = utils.load_and_preprocess_img(CONTENT_IMG)
    # style_image = utils.load_and_preprocess_img(STYLE_IMG)

    # utils.display_image(content_image, "Content Image")
    # utils.display_image(style_image, "Style Image")

    # # # generated_image = tf.Variable(content_image)  # or use a random noise initialization
    # generated_image = tf.Variable(tf.random.normal(content_image.shape, mean=0.5, stddev=0.1)) # Random noise initialization

    # # Example of training loop (simplified)
    # for epoch in range(10):  # Number of epochs
    #     loss = train_step(content_image, style_image, generated_image, content_weight=1e3, style_weight=1e-2)
    #     print(f"Epoch {epoch}, Loss: {loss}")

    #     if epoch % 2 == 0:  # Display every 2 epochs
    #             utils.display_image(generated_image.numpy(), f"Generated Image at Epoch {epoch}")
    # ########################################

    # ########################################
    # # Multiple Training Images case
    # ########################################
    # # Image Augmentation (datagen below)
    # The ImageDataGenerator with augmentation transforms the images randomly during training to create variations, helping the model generalize better.
    # Data augmentation is optional. The model can train and run without it. However, adding it can improve generalization, especially if the dataset is small.
    content_datagen = kp_image.ImageDataGenerator(
        rescale=1./255,             # Normalize pixel values
        rotation_range=40,          # Randomly rotates images by up to 40 degrees
        width_shift_range=0.2,      # Shifts images horizontally by up to 20% of width
        height_shift_range=0.2,     # Shifts images vertically by up to 20% of height
        shear_range=0.2,            # Applies shear transformations
        zoom_range=0.2,             # Zooms in/out randomly
        horizontal_flip=True,       # Flips images horizontally
        fill_mode="nearest"         # Fills missing pixels (after transformations) using nearest neighbors
    )
    style_datagen = kp_image.ImageDataGenerator(
        rescale=1./255,          # Normalize pixel values
        rotation_range=40,       # Same augmentations for style images
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    # datagen.fit(training_images) is only required when using flow() on numpy arrays (e.g., flow(X_train, y_train, batch_size=32)).
    # flow_from_directory() directly loads and processes images from disk, applying augmentation on the fly.

    content_generator = content_datagen.flow_from_directory(CONTENT_DIR,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode=None)  # No labels for style transfer

    style_generator = style_datagen.flow_from_directory(STYLE_DIR,
                                                        target_size=(224, 224),
                                                        batch_size=32,
                                                        class_mode=None)

    # Example of training loop (simplified)
    for epoch in range(100): # Increase the number of epochs for better training

        for content_batch, style_batch in zip(content_generator, style_generator):
            # Assuming content_batch and style_batch are numpy arrays with shape (batch_size, 224, 224, 3)
            # Initialize generated_image with random noise for each batch
            generated_image = tf.Variable(tf.random.normal(content_batch.shape, mean=0.5, stddev=0.1))

            # Compute loss and update generated image
            loss = train_step(content_batch, style_batch, generated_image, content_weight=1e3, style_weight=1e-2)
            print(f"Epoch {epoch}, Loss: {loss}")

            # Display generated image (optional)
            if epoch % 5 == 0:
                utils.display_image(generated_image.numpy(), f"Generated Image at Epoch {epoch}")


main()