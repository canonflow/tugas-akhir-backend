import io
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

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

def preprocessing_pipeline(image, image_size=256, from_bytes=True):
    # TODO: Load image and convert to grayscale
    image = Image.open(io.BytesIO(image)).convert('L') if from_bytes else Image.open(image).convert('L')

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

DATASET_PATH = {
    'train': {
        'positive': 'data/csv/positive_train_pairs.csv',
        'negative': 'data/csv/negative_train_pairs.csv'
    },
    'valid': {
        'positive': 'data/csv/positive_valid_pairs.csv',
        'negative': 'data/csv/negative_valid_pairs.csv'
    },
    'test': {
        'positive': 'data/csv/positive_test_pairs.csv'
    }
}

def create_processed_dataframe_from_csv(dir_type, image_size):
    dir_type = dir_type.lower()

    if dir_type not in ['train', 'valid', 'test']:
        raise Exception("'dir_type' must be either train, valid, or test!")
    
    positive_df = pd.read_csv(DATASET_PATH[dir_type]['positive'])
    negative_df = pd.read_csv(DATASET_PATH[dir_type]['negative'])

    df = pd.concat([positive_df, negative_df], ignore_index=True)
    df = df.sampel(frac=1).reset_index(drop=True)
    print("--- Load the CSV file")

    processed_data = []

    refs = [];
    sketches = [];
    labels = [];

    for idx, row in df.iterrows():
        try:
            ref_image = preprocessing_pipeline(row['Reference Path'], image_size=image_size, from_bytes=False)
            print("    ---- Preprocessing Ref Image DONE")

            sketch_image = preprocessing_pipeline(row['Sketch Path'], image_size=image_size, from_bytes=False)
            print("    ---- Preprocessing Sketch Image DONE")

            label = float(row['Score'])

            refs.append(ref_image)
            sketches.append(sketch_image)
            labels.append(label)

            # processed_data.append({
            #     'ref': ref_image.numpy(),
            #     'sketch': sketch_image.numpy(),
            #     'label': label
            # })
            print("    ---- Appends to each array")
        except Exception as e:
            print(f"    ---- [created_processed_dataframe_from_csv] Error Processing row {idx}: {e}")
    
    print("--- Convert to numpy array")
    refs = np.array(refs)
    sketches = np.array(sketches)
    labels = np.array(labels)

    print("--- Ensure correct data types")
    refs = refs.astype(np.float32)
    sketches = sketches.astype(np.float32)
    labels = labels.astype(np.float32)
    
    return [refs, sketches], labels