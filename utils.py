import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

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
    img = np.squeeze(img, axis=0)
    # img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Ensure range is 0-255

    if img.shape[-1] == 4:
        print('1')
        img = img[..., :3]  # Drop alpha channel if present
    
    if img.dtype != np.uint8:
        print('2')
        img = np.clip(img * 255, 0, 255).astype(np.uint8)  # Ensure 0-255 range
        # matplotlib.pyplot.imshow() expects images in uint8 (0-255) format.

    if img.shape[0] == 3 and img.shape[-1] != 3:
        print('3')
        img = np.transpose(img, (1, 2, 0))  # Ensure channels-last format

    plt.imshow(img)
    plt.title(title)
    plt.show()

def content_loss(content, generated):
    return tf.reduce_mean(tf.square(content - generated))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    gram = tf.matmul(a, a, transpose_a=True)
    return gram

def style_loss(style, generated):
    style_gram = gram_matrix(style)
    generated_gram = gram_matrix(generated)
    return tf.reduce_mean(tf.square(style_gram - generated_gram))

def compute_loss(content_weight, style_weight, content_features, style_features, generated_features):
    # Calculate the content loss
    content_loss_value = content_loss(content_features, generated_features)
    
    # Calculate the style loss
    style_loss_value = style_loss(style_features, generated_features)
    
    # Combine the content and style losses using their weights
    total_loss = content_weight * content_loss_value + style_weight * style_loss_value
    
    return total_loss