import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smp
import math
from time import time
from PIL import Image

training_images_file = open('train-images.idx3-ubyte','rb')
training_images = training_images_file.read()
training_images_file.close()

training_labels_file = open('train-labels.idx1-ubyte', 'rb')
training_labels = training_labels_file.read()
training_labels_file.close()

training_images = bytearray(training_images)
training_images = training_images[16:]

training_labels = bytearray(training_labels)
training_labels = training_labels[8:]

training_images = np.array(training_images).reshape(60000, 784)
training_labels = np.array(training_labels)

# img = Image.fromarray(training_images[5].reshape((28,28)))
# img.show()

testing_images_file = open('t10k-images.idx3-ubyte', 'rb')
testing_images = testing_images_file.read()
testing_images_file.close()

testing_labels_file = open('t10k-labels.idx1-ubyte', 'rb')
testing_labels = testing_labels_file.read()
testing_labels_file.close()

testing_images = bytearray(testing_images)
testing_images = testing_images[16:]

testing_labels = bytearray(testing_labels)
testing_labels = testing_labels[8:]

testing_images = np.array(testing_images).reshape(10000, 784)
testing_labels = np.array(testing_labels)