
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow Version:", tf.__version__)


# ==========================================
# TASK 1: IMPLEMENT SIMPLIFIED ALEXNET
# ==========================================

print("\nTASK 1: ALEXNET ARCHITECTURE")

alexnet = models.Sequential(name="AlexNet")

# Input layer
alexnet.add(layers.Input(shape=(227, 227, 3)))

# First convolution layer
alexnet.add(layers.Conv2D(
    96,
    kernel_size=(11, 11),
    strides=4,
    activation="relu"
))

# First max pooling layer
alexnet.add(layers.MaxPooling2D(
    pool_size=(3, 3),
    strides=2
))

# Second convolution layer
alexnet.add(layers.Conv2D(
    256,
    kernel_size=(5, 5),
    padding="same",
    activation="relu"
))

# Second max pooling layer
alexnet.add(layers.MaxPooling2D(
    pool_size=(3, 3),
    strides=2
))

# Third convolution layer
alexnet.add(layers.Conv2D(
    384,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

# Fourth convolution layer
alexnet.add(layers.Conv2D(
    384,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

# Fifth convolution layer
alexnet.add(layers.Conv2D(
    256,
    kernel_size=(3, 3),
    padding="same",
    activation="relu"
))

# Third max pooling layer
alexnet.add(layers.MaxPooling2D(
    pool_size=(3, 3),
    strides=2
))

# Flatten layer
alexnet.add(layers.Flatten())

# First fully connected layer
alexnet.add(layers.Dense(
    4096,
    activation="relu"
))

# Dropout layer - 50%
alexnet.add(layers.Dropout(0.5))

# Second fully connected layer
alexnet.add(layers.Dense(
    4096,
    activation="relu"
))

# Dropout layer - 50%
alexnet.add(layers.Dropout(0.5))

# Output layer - 10 classes
alexnet.add(layers.Dense(
    10,
    activation="softmax"
))

# Display AlexNet model summary
print("\nAlexNet Model Summary:")
alexnet.summary()

print("\nTask 1 completed successfully!")


# ==========================================
# TASK 2: RESIDUAL BLOCK AND RESNET
# ==========================================

print("\nTASK 2: RESNET ARCHITECTURE")


# Define a residual block
def residual_block(input_tensor, filters):

    # First convolution layer
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu"
    )(input_tensor)

    # Second convolution layer
    x = layers.Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same"
    )(x)

    # Skip connection: Add input to output
    x = layers.Add()([x, input_tensor])

    # Apply activation after addition
    x = layers.Activation("relu")(x)

    return x


# Define input layer
inputs = layers.Input(shape=(64, 64, 3))

# Initial convolution layer
x = layers.Conv2D(
    filters=64,
    kernel_size=(7, 7),
    strides=2,
    padding="same",
    activation="relu"
)(inputs)

# First residual block
x = residual_block(x, filters=64)

# Second residual block
x = residual_block(x, filters=64)

# Flatten layer
x = layers.Flatten()(x)

# Fully connected layer
x = layers.Dense(
    128,
    activation="relu"
)(x)

# Output layer
outputs = layers.Dense(
    10,
    activation="softmax"
)(x)

# Create ResNet-like model
resnet = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="ResNet_Like_Model"
)

# Display model summary
print("\nResNet Model Summary:")
resnet.summary()

print("\nTask 2 completed successfully!")
print("\nQuestion 5 completed successfully!")