# CS5720 - Neural Network & Deep Learning
# Home Assignment 2
# Task 4: CNN Feature Extraction with Filters and Pooling

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

# -------------------------------------------------
# Task 1: Edge Detection Using Sobel Filters
# -------------------------------------------------

# Load the image in grayscale
image = cv2.imread("sample_image.jpg", cv2.IMREAD_GRAYSCALE)

# Check whether the image was loaded correctly
if image is None:
    raise FileNotFoundError(
        "sample_image.jpg was not found. "
        "Make sure the image is inside the Assignment-2 folder."
    )

# Sobel filter for detecting vertical edges
sobel_x_kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Sobel filter for detecting horizontal edges
sobel_y_kernel = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Apply the Sobel filters
sobel_x = cv2.filter2D(
    image,
    cv2.CV_32F,
    sobel_x_kernel
)

sobel_y = cv2.filter2D(
    image,
    cv2.CV_32F,
    sobel_y_kernel
)

# Convert negative values to positive magnitudes for display
sobel_x_display = cv2.convertScaleAbs(sobel_x)
sobel_y_display = cv2.convertScaleAbs(sobel_y)

# Display original, Sobel-X, and Sobel-Y images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(sobel_x_display, cmap="gray")
plt.title("Edge Detection - Sobel X")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(sobel_y_display, cmap="gray")
plt.title("Edge Detection - Sobel Y")
plt.axis("off")

plt.tight_layout()

# Save the figure
plt.savefig(
    "task4_sobel_edges.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -------------------------------------------------
# Task 2: Max Pooling and Average Pooling
# -------------------------------------------------

# Set a seed so the same random matrix can be reproduced
np.random.seed(42)

# Create a random 4x4 matrix
input_matrix = np.random.randint(
    1,
    10,
    size=(4, 4)
).astype(np.float32)

print("\nOriginal 4x4 Matrix:")
print(input_matrix)

# TensorFlow pooling layers expect:
# (batch, height, width, channels)
input_tensor = input_matrix.reshape(
    1, 4, 4, 1
)

# Create a 2x2 max pooling layer
max_pool = tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2),
    strides=2
)

# Create a 2x2 average pooling layer
avg_pool = tf.keras.layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=2
)

# Apply pooling
max_pooled = max_pool(input_tensor)
avg_pooled = avg_pool(input_tensor)

# Remove batch and channel dimensions
max_pooled_matrix = max_pooled.numpy()[0, :, :, 0]
avg_pooled_matrix = avg_pooled.numpy()[0, :, :, 0]

print("\nMax Pooled Matrix:")
print(max_pooled_matrix)

print("\nAverage Pooled Matrix:")
print(avg_pooled_matrix)

print("\nTask 4 completed successfully!")