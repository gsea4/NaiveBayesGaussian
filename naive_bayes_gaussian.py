import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as smp
import math
import random
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

training_images_is_five = training_images[np.where(training_labels == 5)]
training_labels_is_five = training_labels[np.where(training_labels == 5)]

training_images_not_five = training_images[np.where(training_labels != 5)]
training_labels_not_five = training_labels[np.where(training_labels != 5)]

mask_90_percent = np.random.rand(1000) < 0.9

mask_is_five = random.sample(range(training_images_is_five.shape[0]), 1000)
training_images_is_five = training_images_is_five[mask_is_five]
training_images_is_five_set_1 = training_images_is_five[mask_90_percent]
training_labels_is_five = training_labels_is_five[mask_is_five]
training_labels_is_five_set_1 = training_labels_is_five[mask_90_percent]

mask_not_five = random.sample(range(training_images_not_five.shape[0]), 1000)
training_images_not_five = training_images_not_five[mask_not_five]
training_images_not_five_set_2 = training_images_not_five[mask_90_percent]
training_labels_not_five = training_labels_not_five[mask_not_five]
training_labels_not_five_set_2 = training_labels_not_five[mask_90_percent]

mask_10_percent = np.logical_not(mask_90_percent)
testing_images_set_1 = training_images_is_five[mask_10_percent]
testing_images_set_2 = training_images_not_five[mask_10_percent]
testing_labels_set_1 = training_labels_is_five[mask_10_percent]
testing_labels_set_2 = training_labels_not_five[mask_10_percent]

training_images = np.vstack((training_images_is_five_set_1, training_images_not_five_set_2))
training_labels = np.append(training_labels_is_five_set_1, training_labels_not_five_set_2)

testing_images = np.vstack((testing_images_set_1, testing_images_set_2))
testing_labels = np.append(testing_labels_set_1, testing_labels_set_2)
# img = Image.fromarray(training_images[5].reshape((28,28)))
# img.show()