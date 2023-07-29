

import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2
import torch
import numpy as np
from main import get_model

MISCLASSIFIED_IMAGES_DIR = "misclassified_images_dir"

CAMS_DIR = "CAMS_DIR"

# Function to read images from a directory
def read_images_from_directory(directory):
    image_files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    images = [imread(os.path.join(directory, f)) for f in image_files]
    return images

# Function to display images in a grid
def display_images_in_grid(images, rows, cols):
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def apply_grad_cam():
    images = read_images_from_directory(MISCLASSIFIED_IMAGES_DIR)
    CAM_IMAGES_DIR = "cam_dir"
    if not os.path.exists(CAM_IMAGES_DIR):
        os.makedirs(CAM_IMAGES_DIR)
    for index, image in enumerate(images):
        rgb_img = cv2.resize(image, (32, 32))
        rgb_img = np.float32(rgb_img) / 255

        input_tensor = preprocess_image(rgb_img, mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5]).to(device)
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cams = cam(input_tensor=input_tensor, targets=None)
        cam_image = show_cam_on_image(rgb_img, grayscale_cams[0, :], use_rgb=True)
        cv2.imwrite(f'{CAM_IMAGES_DIR}/cam_{index}.jpg', cam_image)

def display_misclassified_images():
    images = read_images_from_directory(MISCLASSIFIED_IMAGES_DIR)
    rows = 2
    cols = 5

    # Display the images in a grid
    display_images_in_grid(images, rows, cols)

def display_cam_images():
    images = read_images_from_directory(CAM_IMAGES_DIR)

    rows = 2
    cols = 5

    # Display the images in a grid
    display_images_in_grid(images, rows, cols)