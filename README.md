# Project Description

In this project we built a Convolutional Neural Network that is capable of recognizing the key facial points, and predict a person's emotions. This project is divided into 3 parts.

### Part A: Detecting key facial points

In part A, we implement a deep neural network model using regression capable of recognizing the key facial points of people's face. It can be divided in the following steps:

  1. **Loading Data:** This part consists of get some information about the dataset.
  2. **Reshaping Data:** Since we will be processing from a one-dimension dataset, we will reshape it into 2D dataset.
  3. **Visualising Data:** Visualizing data ensures the image are reshaped in an appropriate manner.
  4. **Augmenting Data:** It is better to have more data to make a better predicting model. If we don't have additional data, we can perform data augmentation which consist of creating duplicate images and altering them to introduce noise. We will augment the data by duplicating the images from the dataset and doing horizontal flips, increasing brightness and vertical flips.
  5. **Data Normalization:** Normalizing images ensures that pixel values of all images are scaled to a consistent range. This helps with speeding up the convergence of NN.
  6. **Data Preparation:** We used 80% of the database for training and 20% for testing. The images picked for test and training are random.
  7. **Building a Deep ResNet Convolution Block and Identity Blocks:** We implemented ResNet-18 architecture due to its efficiency and efficeness in deep learning tasks. It incorporates residual blocks to acoid the vanishing gradient problem which allows to train deeper networks. Since we are building ResNet-18 by ourself (to show understanding), specific steps in building the model are required. Refer to the code for details.
  8. **Compiling And Training The Model:** This parts consists of training the model, saving the best model, as well as its architecture.
  9. **Testing the Model's Performance:** We evaluated the accuracy. We were able to achieve 90.2% of accuracy for part A.

### Part B: Detecting the facial expression

In part B, we implement a deep neural network model using classification techniques. It is capable of detecting people's expression amongst the following 5 expressions:
  1. Anger
  2. Digust
  3. Sad
  4. Happiness
  5. Surprise

This part is divided into the following steps:

  1. **Loading Data:** This part consists of get some information about the dataset.
  2. **Reshaping Data:** We reshape images of 48 by 48 pixels to 96 by 96 pixels.
  3. **Visualising Data:** We visualize images and their emotion for sanity checks and the number of images per emotions. 
  4. **Data Preparation:** We used 80% of the database for training and 20% for testing. The images picked for test and training are random.
  5. **Data Normalization:** Normalizing images ensures that pixel values of all images are scaled to a consistent range. This helps with speeding up the convergence of NN.
  6. **Augmenting Data:** To augment data, we use datagen from keras library and performation multiple augmentation techniques.
  7. **Building a Deep ResNet:** Using the same convolution and identity blocks defined in Part A, we build our final model for part B.
  8. **Compiling And Training The Model:** This parts consists of training the model, saving the best model, as well as its architecture.
  9. **Testing the Model's Performance:** We evaluated the accuracy. We were able to achieve 91.6% of accuracy for part B. We also made a confusion matrix to demonstrate our data predictions. Most of our predictions aligns with true results. We also compared accuracy of prediction based on the emotions. Disgust performed the worse (as it had the least images).


### Part C: Combining model from part A and part B

This part consists of combining the models designed in part A and part B and visualizing the results.

Here is a sample image of our results:

![image](https://github.com/AA789-ai/EmotionAI/assets/97749196/ed4874ec-2434-44df-8476-aeb9df6f28c6)
