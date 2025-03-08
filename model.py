import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import models, layers, mixed_precision
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

import utils

CONTENT_IMG = 'content.png'  # Update path
STYLE_IMG = 'style.png'  # Update path
CONTENT_DIR = "./content"
STYLE_DIR = "./style"
SAVE_DIR = './generated_images/'

# print("Checking Content Directory:", os.listdir(CONTENT_DIR))
# print("Checking Style Directory:", os.listdir(STYLE_DIR))

def load_vgg19(input_shape=(224, 224, 3)):
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in vgg.layers:
        layer.trainable = False  # Freeze VGG19 layers
    
    # Specify layers for content and style extraction
    content_layers = ['block4_conv2']  # Layer used for content representation
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']  # Layers for style representation
    
    # Create the model for extracting the content and style features
    content_model = models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in content_layers])
    style_model = models.Model(inputs=vgg.input, outputs=[vgg.get_layer(layer).output for layer in style_layers])

    return content_model, style_model

def display_image(image, title="Generated Image"):
    if image.ndim == 4:
        image = image[0]  # Selecting the first image in the batch

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.show()

def save_generated_images(epoch, generated_images): # USE LATER
    batch_size = generated_images.shape[0]  # Get batch size
    
    # Loop through the batch and save each image
    for i in range(batch_size):
        img_to_save = generated_images.numpy()[i]  # Select the i-th image in the batch
        # Create a unique file name for each image
        file_path = os.path.join(SAVE_DIR, f'generated_image_epoch_{epoch}_image_{i}.png')
        
        # Save the individual image
        tf.keras.preprocessing.image.save_img(file_path, img_to_save)

def train_step(content_batch, style_batch, generated_image, content_weight, style_weight, optimizer, content_model, style_model):
    with tf.GradientTape(persistent=True) as tape:
        content_features = content_model(content_batch)
        style_features = style_model(style_batch)
        generated_content_features = content_model(generated_image)
        generated_style_features = style_model(generated_image)
        
        tape.watch(generated_image)  # Explicitly watch generated_image
        loss = utils.compute_loss(content_weight, style_weight, content_features, style_features, [generated_content_features, generated_style_features])

    gradients = tape.gradient(loss, generated_image)
    # for var in tape.watched_variables(): # For debugging
    #     print("Watched variable: ", var)

    if gradients is not None:
        # print("Gradients shape:", gradients.shape)
        # print("Gradients values:", gradients.numpy())
        if tf.reduce_any(tf.math.is_nan(gradients)):
            print("NaN gradients detected!")
            return loss  # Could return early to avoid applying invalid gradients
        else:
            # If the gradients are fine, apply them
            optimizer.apply_gradients(zip(gradients, [generated_image]))
            return loss
    else:
        print("Gradients are None")
        return loss  # Optionally return the loss even if no gradients are calculated

def load_model_weights(model, checkpoint_path):
    # Load the model weights into the model
    model.load_weights(checkpoint_path)

def main():
    # NAME_OF_GEN_IMAGE = ''
    # loaded_image = tf.keras.preprocessing.image.load_img(SAVE_DIR + NAME_OF_GEN_IMAGE)
    # loaded_image = tf.keras.preprocessing.image.img_to_array(loaded_image)  # Convert to numpy array

    scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=5000,
        decay_rate=0.95,
        staircase=True) # Every 5,000 steps learning rate multiplies by 0.95
    
    optimizer = tf.optimizers.Adam(learning_rate=scheduler, clipvalue=1.0) # !Should be reinitialised at every step to handle new var when dealing with many data
    # Restore optimizer state if available
    try:
        checkpoint.restore('optimizer_checkpoint.ckpt')
        print("Optimizer state restored.")
    except:
        print("Optimizer checkpoint not found. Starting fresh.")

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
                                                        class_mode=None,  # No labels for style transfer
                                                        shuffle=False)  # Don't shuffle so we can track the content/style pairing
    style_generator = style_datagen.flow_from_directory(STYLE_DIR,
                                                        target_size=(224, 224),
                                                        batch_size=2,
                                                        class_mode=None,
                                                        shuffle=False)

    # Initialize the generated image once before training loop
    content_batch = next(content_generator)  # Get a batch of content images to determine shape
    content_batch = preprocess_input(content_batch * 255.0)  # Undo rescale=1./255
    style_batch = next(style_generator)
    style_batch = preprocess_input(style_batch * 255.0)

    # generated_image = tf.Variable(preprocess_input(tf.random.uniform(content_batch.shape, minval=0, maxval=255)), trainable=True)
    # Using the code below, for lower loss at the beginning (starting with less noise)
    noise = tf.random.uniform(shape=content_batch.shape, minval=-0.1, maxval=0.1)
    preprocessed_image = preprocess_input(content_batch * 0.5 + style_batch * 0.5 + noise)
    generated_image = tf.Variable(preprocessed_image, trainable=True)

    # Define checkpoint callback to save model weights
    checkpoint_callback = ModelCheckpoint(
        filepath='style_transfer_model_checkpoint.weights.h5',  # Save model to this path
        save_weights_only=True,  # Save only the weights
        save_best_only=True,  # Save the model with the best performance (e.g., lowest loss)
        monitor='loss',  # Monitor loss or other metrics
        mode='min',  # Save when loss is minimized
        verbose=1  # Print status
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer)

    # for content in content_batch:
    #     display_image(content)

    content_images_count = len(content_generator)  # The number of content images
    style_images_count = len(style_generator)      # The number of style images
    content_batch_size = content_generator.batch_size
    style_batch_size = style_generator.batch_size
    steps_per_epoch_content = content_images_count // content_batch_size
    steps_per_epoch_style = style_images_count // style_batch_size

    steps_per_epoch = max(steps_per_epoch_content, steps_per_epoch_style)

    print(f"Steps per epoch: {steps_per_epoch}")

    if os.path.exists('style_transfer_model_checkpoint.weights.h5'):
        print("Loading model weights...")
        load_model_weights(style_model, 'style_transfer_model_checkpoint.weights.h5')

    for epoch in range(70000): # Increase the number of epochs for better training
        print(f"Starting Epoch {epoch}")  # This helps confirm that we enter a new epoch.
    
        for step in range(steps_per_epoch):
        # for step, (content_batch, style_batch) in enumerate(zip(content_generator, style_generator)):
            # Assuming content_batch and style_batch are numpy arrays with shape (batch_size, 224, 224, 3)

            loss = train_step(content_batch, style_batch, generated_image,
                              content_weight=1e2, style_weight=1e-1, optimizer=optimizer,
                              content_model=content_model, style_model=style_model)

            print(f"Epoch {epoch}, Step {step}, Loss: {loss}")

        if epoch % 500 == 0:
            # Optionally save the optimizer state at checkpoints
            optimizer.save('optimizer_checkpoint.ckpt')
            utils.display_image(generated_image.numpy(), f"Generated Image at Epoch {epoch}")
            save_generated_images(epoch, generated_image)

        # print(f"End of Epoch {epoch}, Final Loss: {loss}")
        style_model.save_weights('style_transfer_model_checkpoint.weights.h5')
        checkpoint_callback.on_epoch_end(epoch, logs={'loss': loss}) # Save model weights at the end of each epoch

    checkpoint.save('optimizer_checkpoint.ckpt')


main()