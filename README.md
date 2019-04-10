# CNN-Autoencoder
We are going to work on farsi OCR dataset. As its name implies, it is like famous MNIST dataset but it consists of images of handwritten digits in farsi. Each instance of this dataset is 32 * 32 gray-scale image. It is totally composed of 80000 instances. Train, test sets are loaded using a method in utils.py. Training set includes 0.8 of the whole dataset and the rest is the test set.

The problem we define for this dataset is to reconstruct original image after making some random rotations. We want to develop a model which recieves as input a rotated image and outputs its original without rotation. Meanwhile, a latent embedding is learned in the training process which its quality will be investigated later.
