
# CS5720 - Neural Networks and Deep Learning
# Home Assignment 2 - Question 1
# Student Name: Moulika Reddy
# Task: Text Generation Using LSTM

# Import NumPy for numerical operations
import numpy as np

# Import TensorFlow for building and training the neural network
import tensorflow as tf

# Import Keras layers for building the LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding

# Import the utility for converting text into sequences
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Display the TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Step 2: Load the text dataset

# Create a small text dataset for character-level text generation
text = """
Deep learning is a part of machine learning.
Neural networks learn patterns from data.
Recurrent neural networks are useful for sequential data.
Long short term memory networks can remember information.
Deep learning models can generate new text.
"""

# Convert all characters to lowercase
text = text.lower()

# Display the text dataset
print("\nOriginal Text Dataset:")
print(text)


# Step 3: Convert text into numerical sequences

# Get all unique characters from the dataset
characters = sorted(set(text))

# Count the number of unique characters
vocab_size = len(characters)

# Create a dictionary to convert characters into numbers
char_to_index = {
    char: index for index, char in enumerate(characters)
}

# Create a dictionary to convert numbers back into characters
index_to_char = {
    index: char for index, char in enumerate(characters)
}

# Convert the entire text into a sequence of numbers
encoded_text = [char_to_index[char] for char in text]

# Display the results
print("\nVocabulary Size:", vocab_size)
print("\nUnique Characters:", characters)
print("\nFirst 20 Encoded Characters:", encoded_text[:20])

# Step 4: Prepare input and target sequences

import numpy as np

# Set the number of characters in each input sequence
sequence_length = 40

# Create empty lists for inputs and targets
X = []
y = []

# Create training sequences from the encoded text
for i in range(len(encoded_text) - sequence_length):

    # Select 40 characters as the input sequence
    input_sequence = encoded_text[i:i + sequence_length]

    # Select the next character as the target
    target_character = encoded_text[i + sequence_length]

    X.append(input_sequence)
    y.append(target_character)

# Convert the lists into NumPy arrays
X = np.array(X)
y = np.array(y)

# Display the shape of the training data
print("\nInput Shape:", X.shape)
print("Target Shape:", y.shape)

# Display the first input sequence
print("\nFirst Input Sequence:", X[0])
print("First Target Character:", index_to_char[y[0]])

# Step 5: Build the LSTM model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Create the neural network model
model = Sequential([

    # Convert character numbers into dense vectors
    tf.keras.Input(shape=(sequence_length,)),
    Embedding(input_dim=vocab_size, output_dim=32),

    # LSTM layer learns patterns from character sequences
    LSTM(128),

    # Output layer predicts the next character
    Dense(vocab_size, activation='softmax')

])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Display the model architecture
print("\nLSTM Model Summary:")
model.summary()

# Step 6: Train the LSTM model

print("\nTraining the LSTM model...")

# Train the model using input sequences and target characters
history = model.fit(
    X,
    y,
    epochs=50,
    batch_size=16,
    verbose=1
)

print("\nModel training completed!")

# Step 7: Generate text using the trained LSTM model

def generate_text(seed_text, num_chars=100):

    # Convert the starting text to lowercase
    generated_text = seed_text.lower()

    # Generate one character at a time
    for i in range(num_chars):

        # Take the last 40 characters as input
        input_text = generated_text[-40:]

        # Convert characters into integer values
        encoded_input = [
            char_to_index[char] for char in input_text
        ]

        # Pad shorter inputs to 40 characters
        encoded_input = tf.keras.preprocessing.sequence.pad_sequences(
            [encoded_input],
            maxlen=40,
            padding="pre"
        )

        # Predict the next character
        predictions = model.predict(
            encoded_input,
            verbose=0
        )

        # Select the character with the highest probability
        predicted_index = np.argmax(predictions[0])

        # Convert the predicted index back to a character
        predicted_char = index_to_char[predicted_index]

        # Add the predicted character to the generated text
        generated_text += predicted_char

    return generated_text


# Test the text generation model
seed_text = "deep learning"

print("\nGenerating new text...")

generated_text = generate_text(
    seed_text,
    num_chars=100
)

print("\nGenerated Text:")
print(generated_text)