import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.environ["QT_QPA_PLATFORM"] = "xcb"  # using X11 instead of Wayland in CV2

def preprocess_image(path, size=(64, 64)):
    """Loads and preprocesses an image while blurring the background and keeping the sign sharp."""
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Given path: ({path}) could not be loaded")

    # Resize image
    original_image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    # Convert to grayscale
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Blurring the background
    edges = cv2.Canny(gray, 20, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blurred_image = cv2.GaussianBlur(original_image, (5, 5), 0)
    final_image = np.where(mask_rgb == (255, 255, 255), original_image, blurred_image)
    final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
    
    # adjusting the contrast
    brightness = 4
    contrast = 1.4
    final_image = cv2.addWeighted(final_image, contrast, np.zeros(final_image.shape, final_image.dtype), 0, brightness) 

    # Normalize pixel values
    final_image = final_image / 255.0

    # Add batch dimension
    final_image = np.expand_dims(final_image, axis=0)

    return original_image, final_image
"""
# Testing multiple images
image_paths = [
    "dataset/Train/14/00014_00001_00026.png",
    "dataset/Train/3/00003_00002_00015.png",
    "dataset/Train/16/00016_00000_00005.png",
    "dataset/Train/10/00010_00022_00019.png",
    "dataset/Train/28/00028_00014_00026.png"
]

fig, axes = plt.subplots(len(image_paths), 2, figsize=(10, 3 * len(image_paths)))

for i, path in enumerate(image_paths):
    original, processed_image = preprocess_image(path)

    # Display original image
    axes[i, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
    axes[i, 0].axis("off")
    axes[i, 0].set_title(f"Original Image {i+1}")

    # Display processed image
    axes[i, 1].imshow(processed_image[0])  # Remove batch dimension for display
    axes[i, 1].axis("off")
    axes[i, 1].set_title(f"Processed Image {i+1}")

    print(f"Processed Image {i+1} shape : {processed_image.shape}")

plt.tight_layout()
plt.show()
"""