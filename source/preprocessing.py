import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


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

    # adjusting the contrast
    brightness = 4
    contrast = 1.4
    final_image = cv2.addWeighted(final_image, contrast, np.zeros(final_image.shape, final_image.dtype), 0, brightness) 
    
    # Convert color image to grayscale and extract HOG features
    gray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)  
    final_image = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    
    # Normalize pixel values
    final_image = final_image / 255.0
    
    return final_image
