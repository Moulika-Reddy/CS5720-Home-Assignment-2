# CS5720-Home-Assignment-2

# CS5720 Neural Network & Deep Learning

## Home Assignment 2

**Student Name:** Moulika Reddy Karra
**Student ID:** 700811754
**Course:** CS5720 – Neural Network & Deep Learning
**Semester:** Fall 2026
**University:** University of Central Missouri

---

## Assignment Overview

This assignment demonstrates the implementation of Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) networks, convolution operations, CNN feature extraction, and deep learning architectures using Python, TensorFlow, and Keras.

The assignment consists of five programming questions.

---

## Task 1: Text Generation Using LSTM

**Python File:** `task1_text_generation.py`

### Description

Implemented a character-level text generation model using an LSTM network.

The program performs the following operations:

* Creates a text dataset containing sentences about neural networks and deep learning.
* Converts characters into numerical values.
* Prepares input sequences and target characters.
* Builds an LSTM model using Embedding, LSTM, and Dense layers.
* Trains the model for 50 epochs.
* Generates new text using the trained model.

### Results

The model successfully completed training and generated text based on the input dataset.

The final training accuracy was approximately 98%, although results may vary between runs.

### Output

The terminal output displays the model summary, training accuracy, loss, and generated text.

---

## Task 2: Sentiment Classification Using LSTM

**Python File:** `task2_sentiment_classification.py`

### Description

Implemented an LSTM model to classify IMDB movie reviews as positive or negative.

The program performs the following operations:

* Loads the IMDB dataset containing 25,000 training reviews and 25,000 testing reviews.
* Converts reviews into padded sequences.
* Builds an LSTM model using Embedding, LSTM, and Dense layers.
* Trains the model for 5 epochs.
* Evaluates the model using the testing dataset.
* Generates a confusion matrix and classification report.

### Results

The model achieved approximately 86.05% classification accuracy on the test dataset.

**Confusion Matrix:**

| Actual / Predicted | Negative | Positive |
| ------------------ | -------: | -------: |
| Negative           |   10,641 |    1859 |
| Positive           |    1,628 |   10,872 |

The model correctly classified 21,513 out of 25,000 movie reviews.

### Output
The terminal output displays the training results, test accuracy, confusion matrix, and classification report.

---

## Task 3: Convolution Operations with Different Parameters

**Python File:** `task3_convolution.py`

### Description

Implemented convolution operations on a 5×5 input matrix using a 3×3 kernel with different stride and padding settings.

The following combinations were tested:

| Stride | Padding | Output Shape |
| ------ | ------- | ------------ |
| 1      | VALID   | 3×3          |
| 1      | SAME    | 5×5          |
| 2      | VALID   | 2×2          |
| 2      | SAME    | 3×3          |

### Results

The VALID padding operation reduced the output dimensions, while SAME padding preserved the input dimensions when stride was 1.

Increasing the stride from 1 to 2 reduced the spatial dimensions of the output feature map.

The program successfully generated all four output feature maps.

---

## Task 4: CNN Feature Extraction and Pooling

**Python File:** `task4_cnn_feature_extraction.py`

### Description

Implemented CNN operations using image edge detection and pooling.

The program performs the following operations:

* Loads an input image.
* Applies Sobel X and Sobel Y filters for edge detection.
* Displays the original image and the resulting edge-detection images.
* Performs max pooling and average pooling on a 4×4 matrix.

### Results

The Sobel filters highlighted horizontal and vertical intensity changes in the input image.

The pooling operations reduced the original 4×4 matrix to a 2×2 matrix.

**Max Pooling Output:**

```text
[[7. 8.]
 [6. 8.]]
```

**Average Pooling Output:**

```text
[[5.25 7.  ]
 [4.5  5.75]]
```

The program successfully completed the edge-detection and pooling operations.

---

## Task 5: Implementing and Comparing CNN Architectures

**Python File:** `task5_cnn_architectures.py`

### Part 1: AlexNet Architecture

Implemented a simplified AlexNet model using TensorFlow/Keras.

The architecture includes:

* Five convolution layers.
* Three max pooling layers.
* One flatten layer.
* Two fully connected layers with 4096 neurons each.
* Two dropout layers with a dropout rate of 50%.
* An output layer with 10 neurons and softmax activation.

**Total Parameters:** 58,322,314

The AlexNet model was successfully created, and its model summary was displayed.

### Part 2: Residual Block and ResNet

Implemented a ResNet-like model using residual blocks and skip connections.

The architecture includes:

* An initial convolution layer with 64 filters.
* Two residual blocks.
* Two convolution layers within each residual block.
* Skip connections using Add layers.
* A flatten layer.
* A fully connected layer with 128 neurons.
* An output layer with 10 neurons and softmax activation.

**Total Parameters:** 8,547,210

The ResNet-like model was successfully created, and its model summary was displayed.

### Comparison

AlexNet uses a sequential architecture with convolution, pooling, and fully connected layers.

ResNet uses residual blocks with skip connections that allow information to pass through the network more directly.

Both architectures were successfully implemented using TensorFlow/Keras.

---

## Technologies Used

* Python 3.12
* TensorFlow 2.21.0
* Keras
* NumPy
* Matplotlib
* Scikit-learn
* OpenCV
* Visual Studio Code

---

## Conclusion

All five programming questions were successfully implemented using Python and TensorFlow/Keras.

This assignment helped me understand how RNNs and LSTMs process sequential data, how convolution and pooling operations extract image features, and how AlexNet and ResNet architectures are constructed for image classification.
