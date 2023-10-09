# MNIST-Digit-Classification-using-Deep-Learning
## Task
Build a Machine Learning Model to classify digit using Neural Network Techniques</br>
Tool Used: Google Colab, Python
## Steps Involved
### Import the Dependencies
- Numpy
- Pandas
- Matplotlib
- seaborn
- cv2_imshow from google.colob.patches
- Image from pillow(PIL)
- Tensorflow
- Keras
- MNIST
- Confusion matrix from Tensorflow
### Data Collection and Processing
- Loading the MNIST data from Keras.dataset
- All the images have same dimensions in this dataet, If not, We have to resize all the images to common dimension
- Normalization: Scaling the values, Get all the values to the range of 0 and 1
### Building the Neural Network
- Setting up the layers of Neural Network
  - 1 input layer with flattern function
  - 2 hidden layers of 50 nodes with 'Dense' function and 'relu' activation
  - 1 output layer with 10 nodes with 'Dense' function and 'sigmoid' activation
- Compile the neural network
- Train the build model with training data
  - Training data accuracy = 98.8%
- Get the accuracy on test data
  - Test Data Accuracy = 97.4%
- Build a confusion matrix and heatmap of the confusion matrix
### Building a Predictive System
- Get the input image
- SHow the image using cv2_imshow function
- Covert the RGB image to Grayscale
- Reduce the Size of the image
- Normalize the values
- Resize the numpy image data by increasing 1 dimension
- Predict the input using the build model
- Take the larget probability from the input predict which will be our outcome using argmax function
- Print the result 
