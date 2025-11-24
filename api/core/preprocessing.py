import io
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

def apply_gaussian_blur(image, kernel_size=5, sigma=0.8):
    """
    Applies gentle Gaussian blur to reduce noise before sharpening.
    image: numpy array, float32, [0, 255]
    Returns: blurred image
    """
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(img_uint8, (kernel_size, kernel_size), sigma)
    return blurred.astype(np.float32)

def sharpen_image(image):
    """
    Applies unsharp mask sharpening.
    image: numpy array, float32, [0, 255], shape (H, W, 3)
    Returns: sharpened image
    """
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    img_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    img = cv2.filter2D(img_uint8, -1, kernel)
    return img.astype(np.float32)

def preprocess_for_model(image):
    return tf.keras.applications.efficientnet_v2.preprocess_input(image)

def preprocessing_pipeline(image_bytes, image_size=256):
    # TODO: Load image and convert to grayscale
    image = Image.open(io.BytesIO(image_bytes)).convert('L')

    # TODO: Convert to RGB (duplicate channel)
    image_rgb = image.convert('RGB')
    image_array = np.array(image_rgb, dtype=np.float32)

    # TODO: Apply Gaussian blur
    image_blurred = apply_gaussian_blur(image_array, kernel_size=5, sigma=0)

    # TODO: Apply Sharpening
    image_sharpened = sharpen_image(image_blurred)

    # TODO: Resize
    image_resized = tf.image.resize(image_sharpened, (image_size, image_size))

    # TODO: Preprocess for model
    preprocessed = preprocess_for_model(image_resized)

    return preprocessed
