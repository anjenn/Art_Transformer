import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
from PIL import ImageFilter

# Function to load and process the images
def load_img_normally(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize for VGG model compatibility
    img = np.array(img)
    img = img[tf.newaxis, :]
    return img

def load_and_preprocess_img(img_path):
    # Original image has 4 channels (RGBA), but the model expects 3 channels (RGB).
    
    img = Image.open(img_path)
    print(f"Image mode before conversion: {img.mode}")
    img = img.filter(ImageFilter.GaussianBlur(radius=3))  # Apply Gaussian Blur with radius 2


    img = img.convert("RGBA")  # Ensure it is RGBA before processing
    img = img.convert('RGB')  # Remove the alpha channel

    # Resize the image to 224x224
    img = img.resize((224, 224))  # Resize the image to match model input
    img_array = np.array(img)

    img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
    # print(f"Image array after normalization: {img_array[0, 0]}")  # Show the first pixel value (R, G, B)

    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)

    return img_array

# Display the images
def display_image(img, title="Image"):
    # img = np.squeeze(img, axis=0)
    # img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Ensure range is 0-255

    if img.ndim == 4:  # This means it's a batch of images (batch_size, height, width, channels)
        for i in range(img.shape[0]):  # Loop over the batch
            single_img = img[i]

            # Process each image in the batch
            if single_img.shape[-1] == 4:
                single_img = single_img[..., :3]  # Drop alpha channel if present

            if single_img.dtype != np.uint8:
                single_img = np.clip(single_img * 255, 0, 255).astype(np.uint8)  # Ensure 0-255 range

            if single_img.shape[0] == 3 and single_img.shape[-1] != 3:
                single_img = np.transpose(single_img, (1, 2, 0))  # Ensure channels-last format

            # Display each image from the batch
            plt.imshow(single_img)
            plt.title(f"{title} - Image {i + 1}")
            plt.show()

    elif img.ndim == 3:  # This means it's a single image (height, width, channels)
        # Process a single image
        if img.shape[-1] == 4:
            img = img[..., :3]  # Drop alpha channel if present

        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Ensure 0-255 range

        if img.shape[0] == 3 and img.shape[-1] != 3:
            img = np.transpose(img, (1, 2, 0))  # Ensure channels-last format

        # Display the single image
        plt.imshow(img)
        plt.title(title)
        plt.show()
    else:
        print("Invalid image shape, cannot display.")

def gram_matrix(input_tensor):
    # Ensure input_tensor is a valid tensor
    if len(input_tensor.shape) < 3:
        raise ValueError(f"Expected input tensor to have at least 3 dimensions, but got shape {input_tensor.shape}")
    if not isinstance(input_tensor, tf.Tensor):
        input_tensor = tf.convert_to_tensor(input_tensor)

    # Calculate the gram matrix
    channels = int(input_tensor.shape.as_list()[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    gram = tf.linalg.matmul(a, a, transpose_a=True)  # Use reshaped 'a' for the matrix multiplication
    gram /= tf.cast(tf.size(input_tensor), tf.float32)
    return gram

def style_loss(style_features, generated_features):
    style_features = tf.convert_to_tensor(style_features) if not isinstance(style_features, tf.Tensor) else style_features
    generated_features = tf.convert_to_tensor(generated_features) if not isinstance(generated_features, tf.Tensor) else generated_features

    loss = 0
    for style, generated in zip(style_features, generated_features):
        style_gram = gram_matrix(style)
        generated_gram = gram_matrix(generated)
        loss += tf.reduce_mean(tf.square(style_gram - generated_gram))  # MSE between gram matrices
    
    print(f"Style Loss: {loss}")
    return loss

def content_loss(content_features, generated_features):
    # For each pair of content and generated features, calculate MSE
    loss = 0

    for content, generated in zip(content_features, generated_features):
        loss += tf.reduce_mean(tf.square(content - generated))  # MSE between features
        
    print(f"Content Loss: {loss}")
    return loss

def compute_loss(content_weight, style_weight, content_features, style_features, generated_features):
    [generated_content_features, generated_style_features] = generated_features

    content_loss_value = content_loss(content_features, generated_content_features)

    style_loss_value = style_loss(style_features, generated_style_features)

    # Total Loss
    # print(content_weight, content_loss_value, style_weight, style_loss_value)
    total_loss = content_weight * content_loss_value + style_weight * style_loss_value
    return tf.reduce_sum(total_loss)

def resize_style_features(features):
    TARGET_HEIGHT, TARGET_WIDTH = 224, 224
    
    resized_style_features = []

    for style in features:
        resized_style = tf.image.resize(style, (TARGET_HEIGHT, TARGET_WIDTH))
        
        if resized_style.shape[-1] != 3:
            resized_style = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(resized_style)
        
        resized_style_features.append(resized_style)

    return resized_style_features

def resize_features(features):
    TARGET_HEIGHT, TARGET_WIDTH = 224, 224
    
    resized_features = []

    for feature in features:
        # Resize spatial dimensions (height, width) to match the generated features
        resized_feature = tf.image.resize(feature, (TARGET_HEIGHT, TARGET_WIDTH))
        
        # Adjust the depth by applying a 1x1 convolution if necessary
        if resized_feature.shape[-1] != 3:
            resized_feature = tf.expand_dims(resized_feature, axis=0)  # Add batch dimension
            resized_feature = tf.keras.layers.Conv2D(3, (1, 1), padding="same")(resized_feature)
        
        resized_features.append(resized_feature)

    return resized_features

def global_average_pooling(features):
    return tf.reduce_mean(features, axis=[1, 2])  # Global Average Pooling along height and width
