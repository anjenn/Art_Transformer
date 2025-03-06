import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow import keras
from tensorflow.keras import layers

import utils

CONTENT_PATH = 'content.png'  # Update path
STYLE_PATH = 'style.png'  # Update path

# Adam optimizer to optimize the generated image
optimizer = tf.optimizers.Adam(learning_rate=0.02)

def train_step(content_image, style_image, generated_image, content_weight, style_weight):
    with tf.GradientTape() as tape:
        content_features, style_features = extract_features(content_image, style_image)
        generated_features = model(generated_image)

        # Cost = content + style loss
        loss = utils.compute_loss(content_weight, style_weight, content_features, style_features, generated_features)

    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])

    return loss

# Build the CNN model from scratch
def build_model(input_shape=(224, 224, 3)):
    model = tf.keras.Sequential()
    
    # First few layers (Convolutional layers)
    model.add(layers.InputLayer(input_shape=input_shape))
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

# Function to extract features from the model
def extract_features(content_image, style_image):
    content_features = model(content_image)
    style_features = model(style_image)
    
    return content_features, style_features


def main():

    content_image = utils.load_and_preprocess_img(CONTENT_PATH)
    style_image = utils.load_and_preprocess_img(STYLE_PATH)

    utils.display_image(content_image, "Content Image")
    utils.display_image(style_image, "Style Image")

    # generated_image = tf.Variable(content_image)  # or use a random noise initialization
    
    # Alternative:
    generated_image = tf.Variable(tf.random.normal(content_image.shape, mean=0.5, stddev=0.1)) # Random noise initialization


    # Example of training loop (simplified)
    for epoch in range(10):  # Number of epochs
        loss = train_step(content_image, style_image, generated_image, content_weight=1e3, style_weight=1e-2)
        print(f"Epoch {epoch}, Loss: {loss}")

        if epoch % 2 == 0:  # Display every 2 epochs
                utils.display_image(generated_image.numpy(), f"Generated Image at Epoch {epoch}")

main()