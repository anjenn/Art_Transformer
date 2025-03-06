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

def rgba2rgb(rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    if ch == 3:  # If the image is already RGB, just return it
        return rgba
    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32') # Create an empty RGB image
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3] # Separate the RGBA channels
    a = np.asarray(a, dtype='float32') / 255.0 # Normalize alpha to [0, 1]
    R, G, B = background # Get the background color

    # Apply alpha blending
    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    # Return the result as an RGB image in uint8 format
    return np.asarray(rgb, dtype='uint8')

def load_and_preprocess_img(img_path):
    # Original image has 4 channels (RGBA), but the model expects 3 channels (RGB).
    
    img = Image.open(img_path)
    print(f"Image mode before conversion: {img.mode}")

    img = img.convert("RGBA")  # Ensure it is RGBA before processing
    # img = img.convert('RGB')  # Remove the alpha channel

    img_array = np.array(img)
    img_rgb = rgba2rgb(img_array, background=(255, 255, 255))

    # Resize to 224x224 (or any desired size)
    img_rgb_resized = Image.fromarray(img_rgb).resize((224, 224))

    # Normalize the pixel values
    img_array_normalized = np.array(img_rgb_resized) / 255.0


    # # Resize the image to 224x224
    # img = img.resize((224, 224))  # Resize the image to match model input
    # img_array = np.array(img)   

    # print(f"Image array before normalization: {img_array[0, 0]}")  # Show the first pixel value (R, G, B)

    # img_array = np.array(img) / 255.0  # Normalize the image to [0, 1]
    # print(f"Image array after normalization: {img_array[0, 0]}")  # Show the first pixel value (R, G, B)

    img_array = np.expand_dims(img_array_normalized, axis=0)  # Add batch dimension (1, 224, 224, 3)

    # print(f'Image shape: {img_array.shape}')
    return img_array

# Display the images
def display_image(img, title="Image"):
    img = np.squeeze(img, axis=0)
    img = np.array(img, dtype=np.uint8)
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