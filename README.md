# CIFAR-10 Image Classification with Dense Neural Networks

A deep learning project that classifies images from the CIFAR-10 dataset into 10 categories using a dense (fully connected) neural network built with TensorFlow/Keras.
This project achieves **40.3% accuracy** in 10 epochs  and demonstrates fundamental concepts in image preprocessing, neural network architecture, and why CNNs are superior for image tasks.

## Project Overview

This project implements a feedforward neural network to classify 32×32 color images from the CIFAR-10 dataset into 10 different categories.

- **Dataset:** CIFAR-10 (60,000 color images)
- **Algorithm:** Dense Feedforward Neural Network
- **Framework:** TensorFlow/Keras
- **Classes:** 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Model Type:** Multi-class Classification
- **Training Epochs:** 10
- **Final Test Accuracy:** 40.3%
- **Final Test Loss:** ~2.5

##  Model Performance

### Overall Results
- **Test Accuracy:** 40.3%
- **Test Loss:** ~2.5
- **Performance vs Random:** 4× better than random guessing (10%)
- **Training Time:** ~10 epochs at ~8 seconds per epoch

### Per-Class Accuracy (from Confusion Matrix)

| Class | Name | Accuracy | Performance |
|-------|------|----------|-------------|
| 7 | Horse | 59.0% | ✓ Best |
| 0 | Airplane | 55.0% | ✓ Excellent |
| 8 | Ship | 52.3% | ✓ Very Good |
| 6 | Frog | 47.9% | Good |
| 1 | Automobile | 47.0% | Good |
| 9 | Truck | 45.1% | Good |
| 5 | Dog | 35.1% | Moderate |
| 2 | Bird | 33.4% | Struggling |
| 4 | Deer | 27.0% | Struggling |
| 3 | Cat | 12.8% | ✗ Weakest |

### Key Observations

**What the Model Does Well:**
- Recognizes distinct objects (airplanes, horses, ships) with 50%+ accuracy
- Performs well on vehicles and large animals
- Successfully learns patterns despite flattening

**Common Confusions:**
- **Cat ↔ Dog:** 233 + 129 misclassifications (similar animals)
- **Cat → Frog:** 192 misclassifications
- **Dog → Frog:** 164 misclassifications
- Animals are frequently confused with each other

**Why Confusions Happen:**
- Flattening destroys spatial information (shapes, edges)
- Similar textures and colors between animal classes
- Dense networks can't capture "four legs" or "fur" patterns

## 🎓 Understanding the CIFAR-10 Dataset

### Dataset Structure and Dimensions

The CIFAR-10 dataset consists of color images in pixel form. Each image is **32×32 pixels** with **3 color channels (RGB)**.

#### Dataset Dimensions
Training Set: 50,000 images → Shape: (50000, 32, 32, 3)
Test Set: 10,000 images → Shape: (10000, 32, 32, 3)
Each Image: 32×32 pixels → Shape: (32, 32, 3)
Color Channels: RGB (Red, Green, Blue)


### How Images Are Structured

**Understanding the 4D Array Structure:**

1. **Each Pixel:** Represented as an array of RGB color intensities: `[R, G, B]`
2. **Each Row:** Contains 32 pixels (32 arrays of RGB values)
3. **Each Image:** Made up of 32 rows, giving shape (32, 32, 3) → This is a **3D array**
4. **Training Set:** Contains 50,000 of these images → Shape (50000, 32, 32, 3) → This is a **4D array**

**Breaking it down:**
- 32 rows × 32 columns × 3 color channels = One complete image
- 50,000 images in training set makes it a 4D array
- 10,000 images in test set makes it a 4D array

### Why Flattening is Required

**The Core Problem:**

Dense neural networks can **only take 1D input vectors**. They cannot process 2D or 3D data directly. 
Only advanced architectures like **CNNs** can handle 2D, 3D, or 4D inputs.

**The Solution: Flattening**

Since this project uses dense layers, we must flatten the images from 3D to 1D:

Original Image Shape: (32, 32, 3)
Flattened Shape: (3072,) # 32 × 32 × 3 = 3,072 features

Complete Training Set:
Before Flattening: (50000, 32, 32, 3) [4D array]
After Flattening: (50000, 3072) [2D array]

Complete Test Set:
Before Flattening: (10000, 32, 32, 3) [4D array]
After Flattening: (10000, 3072) [2D array]


### What "2D" and "3D" Mean in This Context

**Important Clarification:**

- **1D:** Each individual flattened image represented as a vector of 3,072 features
- **2D:** The entire dataset of flattened images
- Data is fed **image by image** into the model during training
- Each image must be in 1D form (flattened), and the entire dataset becomes 2D

##  Data Preprocessing Pipeline

### 1. Data Loading
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()


- Automatically downloads CIFAR-10 dataset
- Splits into training (50k) and test (10k) sets

### 2. Data Flattening
X_train_flattened = np.reshape(X_train, (50000, 3072))
X_test_flattened = np.reshape(X_test, (10000, 3072))


**Reason:** Dense neural networks cannot process 2D/3D image data directly. They require 1D input vectors.

### 3. Feature Scaling (Normalization)
X_train_flattened_scaled = X_train_flattened / np.max(X_train_flattened)
X_test_flattened_scaled = X_test_flattened / np.max(X_train_flattened)


**Benefits:**
- Makes the model converge faster (requires fewer epochs)
- Improves accuracy and training stability
- Scales pixel values from [0, 255] to [0, 1] range

**Important Note:** Both train and test sets are scaled using the **training set's maximum value** to maintain consistency.

## Understanding Label Encoding and Loss Functions

### Two Approaches for Multi-Class Classification

When working with neural networks for multi-class classification, you have two options:

| Loss Function | Label Format Required | Example |
|---------------|----------------------|---------|
| `sparse_categorical_crossentropy` | Integers (raw labels) | `[3, 7, 0, 1, 2, ...]` |
| `categorical_crossentropy` | One-hot encoded vectors | `[[0,0,0,1,0,...], [0,0,0,0,0,0,0,1,0,0], ...]` |

### This Project Uses `categorical_crossentropy`

**Why One-Hot Encoding is Needed:**

If you use `categorical_crossentropy` as your loss function, you **must** one-hot encode your labels. This converts each integer label into a vector where only one element is 1 and all others are 0.

y_train_encoded = to_categorical(y_train) # Shape: (50000, 10)
y_test_encoded = to_categorical(y_test) # Shape: (10000, 10)



**Example Transformation:**
Original label: 3
One-hot encoded:
0 1 2 3 4 5 6 7 8 9
↑
Position 3 = 1, rest = 0


**Why This Matters:**
- `sparse_categorical_crossentropy` works with integer labels directly
- `categorical_crossentropy` requires one-hot encoded labels
- Using the wrong format will cause training errors

##  Model Architecture

### Network Structure
Input Layer: 3072 neurons (flattened 32×32×3 image)
↓
Hidden Layer 1: 1024 neurons (ReLU activation)
↓
Hidden Layer 2: 512 neurons (ReLU activation)
↓
Dropout: 0.5 probability (prevents overfitting)
↓
Hidden Layer 3: 256 neurons (ReLU activation)
↓
Dropout: 0.5 probability
↓
Output Layer: 10 neurons (Sigmoid activation - one per class)

text

### Model Configuration
model = keras.Sequential([
keras.layers.Dense(units=1024, activation='relu', input_shape=(3072,)),
keras.layers.Dense(units=512, activation='relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units=256, activation='relu'),
keras.layers.Dropout(0.5),
keras.layers.Dense(units=10, activation='sigmoid')
])

text

### Training Configuration
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Categorical Crossentropy
- **Metrics:** Accuracy
- **Epochs:** 10
- **Dropout Rate:** 0.5 (50% of neurons randomly dropped during training)

### Dropout Explained

**Purpose:** Prevent overfitting
- Randomly drops 50% of neurons during training
- Forces network to learn robust features
- Improves generalization to test data

## Confusion Matrix Analysis

The confusion matrix reveals detailed classification patterns and common errors:

### Strong Performers (Diagonal Values)
- **Horse (7):** 590 correct predictions (59.0%)
- **Airplane (0):** 550 correct (55.0%)
- **Ship (8):** 523 correct (52.3%)
- **Automobile (1):** 470 correct (47.0%)

### Weak Performers
- **Cat (3):** Only 128 correct (12.8%) - Most confused class
- **Deer (4):** 270 correct (27.0%)
- **Bird (2):** 334 correct (33.4%)

### Common Misclassifications
- **Cat → Dog:** 233 misclassifications (similar textures, shapes)
- **Cat → Frog:** 192 misclassifications
- **Dog → Cat:** 129 misclassifications (reciprocal confusion)
- **Dog → Frog:** 164 misclassifications

**Why Animals Are Confused:**
- Similar fur textures and colors
- Flattening loses shape information (ears, tails, body structure)
- Dense networks can't learn spatial patterns like "four legs"

## 🚀 Usage Instructions

### 1. Install Dependencies
pip install tensorflow numpy pandas matplotlib



### 2. Clone Repository
git clone https://github.com/yourusername/CIFAR-10.git
cd CIFAR-10

### 3. Run the Script
python cifar10_classifier.py


Or open the Jupyter Notebook:
jupyter notebook CIFAR-10.ipynb



##  Requirements

tensorflow >= 2.0
numpy >= 1.19
pandas >= 1.0
matplotlib >= 3.0



Create a `requirements.txt`:
tensorflow==2.13.0
numpy==1.24.3
pandas==2.0.3
matplotlib==3.7.2



## Project Structure

cifar10-dense-classification/
├── cifar10_classifier.ipynb # Main Jupyter notebook
├── README.md # This documentation
├── requirements.txt # Python dependencies
├── confusion_matrix.png # Saved confusion matrix visualization
└── models/
└── trained_model.h5 # Saved model (optional)



## Why Dense Networks Instead of CNNs?

**Educational Purpose:**

This project deliberately uses dense layers to:
- Demonstrate fundamental preprocessing concepts
- Show why data flattening is necessary for dense layers
- Highlight the importance of proper input formatting
- Understand the limitations of dense networks on image data
- **Prove why CNNs are necessary for computer vision tasks**

### Performance Comparison

| Aspect | Dense Network (This Project) | CNN |
|--------|------------------------------|-----|
| Input handling | Requires flattening to 1D | Handles 2D/3D directly |
| Spatial features | Lost during flattening | Preserved |
| Parameters | Very high (~3M+) | Lower (~100K) |
| Accuracy on CIFAR-10 | **40.3%** (achieved) | 85-95% |
| Training time | Moderate | Similar/Faster |
| Why the difference? | No spatial awareness | Learns edges, shapes, textures |

### Why 40.3% is Actually Good

**Context:**
- **Random guessing:** 10% accuracy
- **This model:** 40.3% accuracy → **4× better than random!**
- **Typical dense networks on CIFAR-10:** 40-55%
- **Simple CNNs:** 70-80%
- **Advanced CNNs:** 85-95%

**What This Proves:**
- Dense networks CAN learn from images
- But they're limited by spatial information loss
- This is exactly why CNNs were invented for computer vision

### Why Normalization is Important

**Without normalization:**
- Pixel values range from 0 to 255
- Large input values slow down learning
- Model takes many epochs to converge
- Gradients can explode or vanish

**With normalization:**
- Pixel values scaled to [0, 1]
- Faster convergence
- Better gradient flow during backpropagation
- Improved final accuracy
- More stable training
