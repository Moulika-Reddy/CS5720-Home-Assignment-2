
# CS5720 - Neural Network & Deep Learning
# Home Assignment 2
# Task 2: Sentiment Classification Using RNN

# Step 1: Import the required libraries

import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Display TensorFlow version
print("TensorFlow Version:", tf.__version__)


# Step 2: Load the IMDB movie review dataset

# Use the 10,000 most frequently occurring words
vocab_size = 10000

# Load training and testing datasets
(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=vocab_size
)

# Display the dataset information
print("\nIMDB Dataset Loaded Successfully!")

print("Number of training reviews:", len(X_train))
print("Number of testing reviews:", len(X_test))

print("\nFirst review (encoded):")
print(X_train[0][:30])

print("\nSentiment of first review:", y_train[0])

# Display the number of positive and negative reviews
print("\nTraining Dataset Distribution:")

print("Positive reviews:", np.sum(y_train == 1))
print("Negative reviews:", np.sum(y_train == 0))


# Step 3: Preprocess the IMDB dataset using padding

# Set the maximum length of each movie review
max_length = 200

# Pad training reviews to the same length
X_train = pad_sequences(
    X_train,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)

# Pad testing reviews to the same length
X_test = pad_sequences(
    X_test,
    maxlen=max_length,
    padding="pre",
    truncating="pre"
)

# Display the shapes after padding
print("\nAfter Padding:")

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Display the first padded review
print("\nFirst padded review:")
print(X_train[0])

# Display the length of the first review
print("\nLength of first padded review:", len(X_train[0]))

# Display the sentiment label
print("Sentiment of first review:", y_train[0])


# Step 4: Build the LSTM model for sentiment classification

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense

# Create the LSTM model
model = Sequential([
    
    # Input layer: Each review contains 200 tokens
    Input(shape=(max_length,)),

    # Embedding layer: Convert word IDs into dense vectors
    Embedding(input_dim=vocab_size, output_dim=32),

    # LSTM layer: Learn patterns in movie reviews
    LSTM(64),

    # Output layer: Predict positive or negative sentiment
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display the model architecture
print("\nLSTM Model Summary:")
model.summary()


# Step 5: Train the LSTM model

print("\nTraining the LSTM model...")

history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

print("\nModel training completed!")


# Step 6: Evaluate the trained LSTM model

print("\nEvaluating the model on test data...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Results:")
print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy, 4))


# Step 7: Generate predictions and classification report

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)

import matplotlib.pyplot as plt

# Predict sentiment probabilities for the test reviews
y_pred_prob = model.predict(X_test)

# Convert probabilities into binary predictions
# Probability >= 0.5 means positive sentiment (1)
# Probability < 0.5 means negative sentiment (0)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Generate the classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Negative", "Positive"],
        digits=4
    )
)

# Calculate the overall classification accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nOverall Classification Accuracy:", round(accuracy, 4))


# Step 8: Plot the confusion matrix

plt.figure(figsize=(7, 5))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix - IMDB Sentiment Classification")

plt.colorbar()

# Add the number of predictions inside each cell
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

# Label the axes
plt.xticks([0, 1], ["Negative", "Positive"])
plt.yticks([0, 1], ["Negative", "Positive"])

plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()

# Save the confusion matrix image
plt.savefig("task2_confusion_matrix.png", dpi=300)

# Display the confusion matrix
plt.show()

print("\nConfusion matrix saved successfully!")