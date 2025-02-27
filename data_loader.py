import os
import random
import numpy as np
from PIL import Image
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def file_counts(path, folder):
    full_path = os.path.join(path, folder)
    count = len(os.listdir(full_path))
    return count

def resize_image(input_path: str, output_path: str, size: tuple = (224, 224)):
    """
    Resizes an image to the specified size and saves it to output path.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the resized image.
        size (tuple(int, int)): The dimensions to resize the image to.
    """
    try:
        with Image.open(input_path) as img:
            img = img.resize(size, Image.Resampling.LANCZOS)
            img.save(output_path)
    except Exception as e:
        print(f"Error resizing {input_path}: {e}")

def augment_images(input_dir, datagen, output_dir, amount=0.15):
    """
    Augments a percentage of images in each class and saves them to the output directory.

    Args:
        input_dir (str): Directory containing images organized by class.
        datagen (ImageDataGenerator): Data augmentation generator.
        output_dir (str): Directory to save augmented images.
        amount (float): Percentage of images to augment.
    """
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Create class-specific output directory
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # List all images in the class
        class_images = os.listdir(class_dir)
        num_to_augment = max(1, int(len(class_images) * amount))
        selected_images = random.sample(class_images, num_to_augment)

        # Copy original images
        for file in class_images:
            original_path = os.path.join(class_dir, file)
            new_path = os.path.join(class_output_dir, file)
            if not os.path.exists(new_path):
                os.link(original_path, new_path)

        # Augment selected images
        for file in class_images:
            input_path = os.path.join(class_dir, file)
            output_path = os.path.join(class_output_dir, file)

            if file in selected_images:
                # Augment and save the image
                with Image.open(input_path) as img:
                    img_array = np.array(img)[np.newaxis, ...]
                    for augmented_img in datagen.flow(img_array, batch_size=1):
                        augmented_img = Image.fromarray(augmented_img[0].astype('uint8'))
                        augmented_img.save(output_path)  # Save augmented image
                        break
            else:
                # Copy the original image if not selected for augmentation
                if not os.path.exists(output_path):
                    os.link(input_path, output_path)

def balance_classes(data, output_dir: str):
    """
    Balances classes by selecting the same number of samples for each class and saves them.

    Args:
        data (list): List of tuples (image array, class label).
        output_dir (str): Directory to save balanced images.

    Returns:
        None
    """
    class_counts = Counter([label for _, label in data])
    min_samples = min(class_counts.values())

    os.makedirs(output_dir, exist_ok=True)

    for cls in class_counts.keys():
        class_samples = [item for item in data if item[1] == cls]
        selected_samples = random.sample(class_samples, min_samples)

        class_dir = os.path.join(output_dir, cls)
        os.makedirs(class_dir, exist_ok=True)

        for i, (img_array, _) in enumerate(selected_samples):
            img = Image.fromarray((img_array * 255).astype('uint8'))
            img.save(os.path.join(class_dir, f"balanced_{i}.png"))

def load_processed_data(input_dir: str, output_dir: str = "./processed_data", augment_chance: float = 0.15, balance: bool = True):
    """
    Fully processes data: resizing, normalization, balancing classes, and augmentation.

    Args:
        input_dir (str): Input directory containing raw images organized by class.
        output_dir (str): Directory to save processed data.
        augment_chance (float): Percentage of images to augment.
        balance (bool): Whether to balance classes. Defaults to True

    Returns:
        tuple: Normalized balanced X (features) and y (labels).
    """
    os.makedirs(output_dir, exist_ok=True)
    classes = ['hatchback', 'moped', 'pickup', 'seden', 'suv']

    # Temporary directories
    resized_dir = os.path.join(output_dir, "resized_data")
    balanced_dir = os.path.join(output_dir, "balanced_data") if balance else resized_dir
    augmented_dir = os.path.join(output_dir, "augmented_data")

    os.makedirs(resized_dir, exist_ok=True)
    os.makedirs(augmented_dir, exist_ok=True)

    if balance:
        os.makedirs(balanced_dir, exist_ok=True)

    # resize images
    for root, _, files in os.walk(input_dir):
        for file in files:
            input_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, input_dir)
            output_class_dir = os.path.join(resized_dir, relative_path)

            os.makedirs(output_class_dir, exist_ok=True)
            output_path = os.path.join(output_class_dir, file)
            
            resize_image(input_path, output_path)
    
    # debugging
    for class_name in classes:
        print(f"resized file count ({class_name}) : {file_counts('processed_data/resized_data', class_name)}")


    # add resized images to data
    data = []
    for class_name in os.listdir(resized_dir):
        class_dir = os.path.join(resized_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            input_path = os.path.join(class_dir, file)
            with Image.open(input_path) as img:
                img_array = np.array(img).astype('float32') / 255.0
                data.append((img_array, class_name.lower()))

    # balance classes if True
    if balance:
        balance_classes(data, balanced_dir)
        for class_name in classes:
            print(f"balanced file count ({class_name}) : {file_counts('processed_data/balanced_data', class_name)}")
    else:
        balanced_dir = resized_dir

    # use data augmentation to increase data diversity (by augment_chance)
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    augment_images(balanced_dir, datagen, augmented_dir, amount=augment_chance)

    # debugging
    for class_name in classes:
        print(f"augmented file count ({class_name}) : {file_counts('processed_data/augmented_data', class_name)}")

    # load final data into X, y and return
    final_data = []
    for class_name in os.listdir(augmented_dir):
        class_dir = os.path.join(augmented_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for file in os.listdir(class_dir):
            input_path = os.path.join(class_dir, file)
            with Image.open(input_path) as img:
                img_array = np.array(img).astype('float32') / 255.0
                final_data.append((img_array, class_name.lower()))

    X = np.array([item[0] for item in final_data])
    y = np.array([item[1] for item in final_data])

    # Bit of cleaning up, used for when we re-run the code later in the notebook
    try: 
        shutil.rmtree(output_dir)
        print(f"Deleted processed_data")
    
    except Exception as e:
        print("Error deleting processed_data")


    return X, y
