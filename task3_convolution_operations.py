
# CS5720 - Home Assignment 2
# Task 3: Convolution Operations with Different Parameters

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

# Step 1: Define the 5x5 input matrix
input_matrix = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

print("Input Matrix:")
print(input_matrix)


# Step 2: Define the 3x3 kernel
kernel = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float32)

print("\nKernel:")
print(kernel)


# Step 3: Reformat the input and kernel
# Conv2D expects input shape:
# (batch_size, height, width, channels)

input_tensor = input_matrix.reshape(1, 5, 5, 1)

# Kernel shape:
# (kernel_height, kernel_width, input_channels, filters)

kernel_tensor = kernel.reshape(3, 3, 1, 1)


# Step 4: Define a function to perform convolution

def perform_convolution(stride, padding):

    # Create the convolution layer
    conv_layer = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        strides=(stride, stride),
        padding=padding.lower(),
        use_bias=False
    )

    # Build the layer using the input tensor
    conv_layer.build(input_tensor.shape)

    # Set the given kernel weights
    conv_layer.set_weights([kernel_tensor])

    # Perform convolution
    output = conv_layer(input_tensor)

    # Convert output tensor to NumPy array
    output_matrix = output.numpy()[0, :, :, 0]

    # Print the results
    print("\n--------------------------------")
    print(f"Stride = {stride}, Padding = {padding}")
    print("--------------------------------")

    print("Output Shape:", output_matrix.shape)

    print("Output Feature Map:")
    print(output_matrix)

    return output_matrix


# Step 5: Perform convolution for all four cases

# Case 1: Stride = 1, Padding = VALID
output1 = perform_convolution(1, "VALID")

# Case 2: Stride = 1, Padding = SAME
output2 = perform_convolution(1, "SAME")

# Case 3: Stride = 2, Padding = VALID
output3 = perform_convolution(2, "VALID")

# Case 4: Stride = 2, Padding = SAME
output4 = perform_convolution(2, "SAME")


print("\nAll convolution operations completed successfully!")