import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import models, layers, mixed_precision

import utils

CONTENT_IMG = 'content.png'  # Update path
STYLE_IMG = 'style.png'  # Update path
CONTENT_DIR = "./content"
STYLE_DIR = "./style"

print("Checking Content Directory:", os.listdir(CONTENT_DIR))
print("Checking Style Directory:", os.listdir(STYLE_DIR))

def load_vgg19(input_shape=(224, 224, 3)):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    
    # Specify layers for content and style extraction
    content_layers = ['block4_conv2']  # Layer used for content representation
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']  # Layers for style representation
    
    # Create the model for extracting the content and style features
    content_model = models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in content_layers])
    style_model = models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in style_layers])

    return content_model, style_model

def train_step(content_batch, style_batch, generated_image, content_weight, style_weight, optimizer, content_model, style_model):
    content_features, style_features = extract_features(content_batch, style_batch, content_model, style_model)
    generated_features, generated_features_ = extract_features(generated_image, generated_image, content_model, style_model)
    content_features, style_features, generated_features = utils.resize_features(content_features, style_features, generated_features)

    height, width, channels = 224, 224, 3  # example image shape (height, width, channels)
    batch_size = tf.shape(content_batch)[0]  # Use the batch size from the actual batch
    initial_value = tf.random.normal([batch_size, height, width, channels], mean=0.5, stddev=1.0)
    generated_image = tf.Variable(initial_value, trainable=True)
    
    with tf.GradientTape() as tape:
        tape.watch(generated_image)  # Explicitly watch generated_image
        loss = utils.compute_loss(content_weight, style_weight, content_features, style_features, generated_features)

    gradients = tape.gradient(loss, generated_image)

    if gradients is not None:
        if tf.reduce_any(tf.math.is_nan(gradients)):
                print("NaN gradients detected")
        else:
            # Apply gradients to model's trainable variables
            optimizer.apply_gradients(zip(gradients, [generated_image]))
            return loss
    else:
        print("Gradients are None")

# Function to extract features from the model
def extract_features(content_image, style_image, content_model, style_model):
    # Get content features using VGG19
    content_features = content_model(content_image)
    
    # Get style features using VGG19
    style_features = style_model(style_image)
    
    return content_features, style_features


def main():
    optimizer = tf.optimizers.Adam(learning_rate=0.001, clipvalue=1.0) # !Should be reinitialised at every step to handle new var when dealing with many data
    content_model, style_model = load_vgg19()

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
    content_generator = content_datagen.flow_from_directory(CONTENT_DIR,
                                                        target_size=(224, 224),
                                                        batch_size=2,
                                                        class_mode=None)  # No labels for style transfer

    style_generator = style_datagen.flow_from_directory(STYLE_DIR,
                                                        target_size=(224, 224),
                                                        batch_size=2,
                                                        class_mode=None)

    # Initialize the generated image once before training loop
    content_batch = next(content_generator)  # Get a batch of content images to determine shape
    generated_image = tf.Variable(tf.random.normal(content_batch.shape, mean=0.5, stddev=0.1))  # Random noise initialization

    # Ensure we have the correct number of steps per epoch
    steps_per_epoch = max(len(content_generator), len(style_generator))
    print(f"Steps per epoch: {steps_per_epoch}")

    # Example of training loop (simplified)
    for epoch in range(7): # Increase the number of epochs for better training
        print(f"Starting Epoch {epoch}")  # This helps confirm that we enter a new epoch.

        for step, (content_batch, style_batch) in enumerate(zip(content_generator, style_generator)):
            # Assuming content_batch and style_batch are numpy arrays with shape (batch_size, 224, 224, 3)

            # Compute loss and update generated image

            try:
                content_batch = next(content_generator)
            except StopIteration:
                content_generator.reset()
                content_batch = next(content_generator)

            try:
                style_batch = next(style_generator)
            except StopIteration:
                style_generator.reset()
                style_batch = next(style_generator)

            loss = train_step(content_batch, style_batch, generated_image,
                              content_weight=1e3, style_weight=1e-2, optimizer=optimizer,
                              content_model=content_model, style_model=style_model)

            print(f"Epoch {epoch}, Step {step}, Loss: {loss}")
            
            if step == 1000:
                utils.display_image(generated_image.numpy(), f"Generated Image at epoch: {epoch}, step: {step}")

            if step == 1500:
                utils.display_image(generated_image.numpy(), f"Generated Image at epoch: {epoch}, step: {step}")

            if step == 2000:
                utils.display_image(generated_image.numpy(), f"Generated Image at epoch: {epoch}, step: {step}")
        print(f"End of Epoch {epoch}, Final Loss: {loss}")

            # Display generated image (optional)
        if epoch % 5 == 0:
            utils.display_image(generated_image.numpy(), f"Generated Image at Epoch {epoch}")


main()