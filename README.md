# Autoencoders

# MNIST Convolutional Autoencoder using TensorFlow/Keras

This project implements a **Convolutional Autoencoder** using TensorFlow and Keras to compress and reconstruct handwritten digit images from the [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database).

## 🧠 Project Overview

Autoencoders are unsupervised neural networks designed to learn compressed representations of input data. In this project, we use a **convolutional architecture** for more effective learning from image data like MNIST.

### Key Features
- 🧱 Encoder and decoder built with `Conv2D` and `Conv2DTranspose` layers
- 🧠 Learns a compressed 64-dimensional latent representation
- 🖼 Reconstructs 28x28 grayscale digit images from the latent space
- 📊 Visualizes original vs. reconstructed images

## 🚀 Getting Started

### Prerequisites

Ensure Python 3.7+ is installed. Then install the required libraries:

### 🧱 Model Architecture

🔒 Encoder

```

encoder = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu')  # Encoded representation
])

```

🔓 Decoder

```
decoder = models.Sequential([
    layers.InputLayer(input_shape=(64,)),
    layers.Dense(7 * 7 * 64, activation='relu'),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

```

🔁 Autoencoder

```
decoder = models.Sequential([
    layers.InputLayer(input_shape=(64,)),
    layers.Dense(7 * 7 * 64, activation='relu'),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

```
### 🖼 Output Visualization
The script visualizes results by showing a grid of original and reconstructed images after training. This helps assess the quality of the learned representations.

📦 Requirements

```
tensorflow>=2.0
numpy
matplotlib
```

### 🧪 Training Details
Dataset: MNIST (60,000 training, 10,000 test images)

Loss Function: Binary Crossentropy

Optimizer: Adam

Epochs: Customize as needed

Image Shape: 28x28 grayscale (normalized to [0, 1])

### 📜 License
This project is licensed under the [MIT](https://opensource.org/license/MIT) License.
