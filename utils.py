import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

# Function to load and process the images
def load_img(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))  # Resize for VGG model compatibility
    img = np.array(img)
    img = img[tf.newaxis, :]
    return img

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