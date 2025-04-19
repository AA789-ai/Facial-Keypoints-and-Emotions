# Facial Keypoint Detection and Emotion Recognition Project

## Project Overview

This project focuses on building and evaluating Convolutional Neural Network (CNN) models for analyzing human faces. It consists of two primary tasks:
1.  **Facial Keypoint Detection:** Identifying the coordinates of key facial features (like eyes, nose, mouth corners) using a regression model.
2.  **Emotion Recognition:** Classifying the facial expression into one of five categories (Anger, Disgust, Sadness, Happiness, Surprise) using a classification model.

Finally, the project demonstrates how to combine the outputs of these two models for a comprehensive facial analysis visualization.

## Key Features

* **Facial Keypoint Detection:** Predicts the location of key facial points using a CNN-based regression approach, evaluated using Root Mean Squared Error (RMSE).
* **Emotion Recognition:** Classifies facial expressions into 5 distinct categories, evaluated using the F1-score metric suitable for potentially imbalanced classes.
* **Custom ResNet Architecture:** Implements a ResNet-18 inspired architecture from scratch for both tasks, demonstrating an understanding of residual blocks.
* **Data Augmentation:** Employs various techniques (flipping, brightness adjustment, Keras `ImageDataGenerator`) to enhance model robustness and performance.
* **Combined Visualization:** Integrates keypoint predictions and emotion classification onto input images.

## Results

* **Keypoint Detection:** The regression model achieved an **RMSE of ~2.01 pixels** on the test set for predicting facial keypoint locations. Lower RMSE indicates better performance.
* **Emotion Recognition:** The classification model reached a **Weighted Average F1-score of 90%** on the test set (Accuracy: 90%). The confusion matrix and classification report showed strong overall performance, though 'Disgust' was less accurately predicted due to fewer training examples.
* **Combined Output:** The system successfully integrates both predictions, providing a visual representation of keypoints and emotion on facial images.

**Sample Combined Output:**

![Facial Analysis Result](https://github.com/AA789-ai/EmotionAI/assets/97749196/ed4874ec-2434-44df-8476-aeb9df6f28c6)
*Example output showing detected facial keypoints (blue dots) and the predicted emotion label overlaid on the input image.*

**Emotion Recognition Performance (Confusion Matrix):**

![Confusion Matrix for Emotion Recognition](https://github.com/user-attachments/assets/9a7ae36c-eaa6-4a51-95fd-9d59207c183c)
*Confusion matrix showing the performance of the 5-class emotion recognition model on the test set. Rows represent true labels, columns represent predicted labels.*

## Technical Details

* **Frameworks/Libraries:** TensorFlow, Keras, OpenCV, Pandas, NumPy, Matplotlib
* **Model Architecture:** Custom ResNet-18 implementation with Convolutional and Identity Blocks.
* **Techniques:** Convolutional Neural Networks (CNNs), Regression, Classification, Data Augmentation, Transfer of architectural components (ResNet blocks) between tasks.

## Methodology

The project is structured into three main parts:

### Part A: Facial Keypoint Detection (Regression)

1.  **Loading Data & Reshaping:**
    * Load the keypoint dataset (images and coordinates).
    * Reshape image data from flattened arrays to 2D image format
2.  **Visualising Data:**
    * Visualize sample images with keypoints to verify loading and reshaping.
3.  **Augmenting Data:**
    * Generate augmented data from the training set by applying transformations like horizontal flipping, vertical flipping (use with caution for faces), and brightness adjustments.
4.  **Data Normalization:**
    * Normalize pixel values (e.g., scaling to [0, 1] or [-1, 1]) to aid model convergence.
5.  **Data Preparation:**
    * Split the data into training (80%) and testing (20%) sets randomly.
6.  **Building Model (Custom ResNet-18):**
    * Implement custom Convolutional and Identity blocks based on the ResNet philosophy to mitigate vanishing gradients in deep networks.
    * Construct the ResNet-18 architecture using these blocks, ending with a regression output layer suitable for predicting keypoint coordinates.
7.  **Compiling And Training The Model:**
    * Compile the model with an appropriate optimizer (e.g., Adam) and loss function for regression (e.g., Mean Squared Error - MSE).
    * Train the model on the training and augmented data, saving the best model weights based on validation performance.
8.  **Testing the Model's Performance:**
    * Evaluate the model on the test set using **Root Mean Squared Error (RMSE)** as the primary metric. The achieved RMSE was approximately **2.01 pixels**.

### Part B: Emotion Recognition (Classification)

1.  **Loading Data:**
    * Load the emotion dataset (images and labels).
2.  **Reshaping Data:**
    * Resize images (e.g., from the original 48x48 pixels to 96x96 pixels to match the expected input size or improve feature extraction.
3.  **Visualising Data:**
    * Visualize sample images per emotion category and check class distribution.
4.  **Data Preparation:**
    * Split the data into training (80%) and testing (20%) sets randomly.
5.  **Data Normalization:**
    * Normalize pixel values.
6.  **Augmenting Data:**
    * Utilize Keras' `ImageDataGenerator` (or similar) to apply on-the-fly augmentation during training (e.g., rotations, shifts, shear, zoom, flips).
7.  **Building Model (Custom ResNet):**
    * Adapt the ResNet-18 architecture defined in Part A for classification. Reuse the Convolutional and Identity blocks.
    * Replace the final layer with a classification head (e.g., Dense layer with Softmax activation for 5 emotion classes).
8.  **Compiling And Training The Model:**
    * Compile the model with an optimizer, a suitable loss function for multi-class classification (e.g., Categorical Crossentropy), and relevant metrics (including F1-score, accuracy).
    * Train the model using the data generator for augmentation. Save the best model based on validation performance.
9.  **Testing the Model's Performance:**
    * Evaluate the model on the test set. The primary metric reported is the **Weighted Average F1-score**, which was **90%**, accounting for class imbalance. The overall accuracy was also 90%.
    * Generate and analyze a confusion matrix and classification report to understand performance across different emotion classes (noting lower performance for the under-represented 'Disgust' class).

### Part C: Combining Models from Part A and Part B

1.  **Model Integration:** Load the trained models from Part A and Part B.
2.  **Visualization Pipeline:** Create a process that takes an input face image, runs inference using both models (predicts keypoints and emotion), and displays the input image with the predicted facial keypoints overlaid and the predicted emotion label annotated.

