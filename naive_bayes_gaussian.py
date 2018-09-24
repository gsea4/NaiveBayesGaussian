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
training_labels_is_five = training_labels[np.where(training_labels == 5)].reshape(training_images_is_five.shape[0],1)
training_images_is_five = np.column_stack((training_images_is_five, training_labels_is_five))

training_images_not_five = training_images[np.where(training_labels != 5)]
training_labels_not_five = training_labels[np.where(training_labels != 5)].reshape(training_images_not_five.shape[0],1)
training_images_not_five = np.column_stack((training_images_not_five, training_labels_not_five))

mask_90_percent = np.random.rand(2000) < 0.9
mask_10_percent = np.logical_not(mask_90_percent)

mask_is_five = random.sample(range(training_images_is_five.shape[0]), 1000)
training_images_is_five = training_images_is_five[mask_is_five]

mask_not_five = random.sample(range(training_images_not_five.shape[0]), 1000)
training_images_not_five = training_images_not_five[mask_not_five]

temp_training_images = np.vstack((training_images_is_five, training_images_not_five))
np.random.shuffle(temp_training_images)

training_images = temp_training_images[mask_90_percent]
testing_images = temp_training_images[mask_10_percent]

training_labels = training_images[:, 784]
training_images = training_images[:, :784]

testing_labels = testing_images[:, 784]
testing_images = testing_images[:, :784]

# img = Image.fromarray(training_images[5].reshape((28,28)))
# img.show()
training_labels[training_labels != 5] = 9
unique_elem, counts = np.unique(training_labels, return_counts = True)
priors = np.append(unique_elem.reshape(2,1), counts.reshape(2,1), 1)

temp = np.array(priors[0])
priors[0] = priors[1]
priors[1] = temp

label_5_summary = np.array((-9,-9))
label_9_summary = np.array((-9,-9))

v0 = np.var(training_images[training_labels != 5])
v1 = np.var(training_images[training_labels == 5])

training_images_is_five = training_images[training_labels == 5]
training_images_not_five = training_images[training_labels != 5]

for pixel in range(28*28):
    m5 = np.mean(training_images_is_five[:, pixel])
    v5 = np.var(training_images_is_five[:, pixel])
    label_5_summary = np.vstack((label_5_summary, np.array((m5, v5))))

    m9 = np.mean(training_images_not_five[:, pixel])
    v9 = np.var(training_images_not_five[:, pixel])
    label_9_summary = np.vstack((label_9_summary, np.array((m9, v9))))
    
label_5_summary = label_5_summary[1:, :]
label_9_summary = label_9_summary[1:, :]

def pdf(x, mean, var):
    exp = math.exp(-(math.pow(x - mean, 2)/(2 * var)))
    pdf = (1 / math.sqrt(2 * math.pi * var)) * exp
    return pdf

count = 0
def classify(test_set, priors, summary_5, summary_9):
    prediction = []
    for image in test_set:
        max_pros_class = (-math.inf, -1)
        for label in range(2):
            log_prior = math.log(priors[label][1])
            p = log_prior
            for pixel in range(28 * 28):
                if label == 0:
                    mean = summary_9[pixel][0]
                    var = v0
                else:
                    mean = summary_5[pixel][0]
                    var = v1

                if pdf(image[pixel],mean,var) > 0:
                    p += math.log(pdf(image[pixel],mean,var))
            
            if p > max_pros_class[0]:
                max_pros_class = (p, label)

        prediction.append(max_pros_class)   
    prediction = np.array(prediction, dtype = 'int16')
    return prediction[:,1]

result = classify(testing_images, priors, label_5_summary, label_9_summary)

correct = 0
testing_labels[testing_labels != 5] = 0
testing_labels[testing_labels == 5] = 1

for i in range(testing_labels.shape[0]):
    if result[i] == testing_labels[i]:
        correct += 1

print(correct)
print(testing_images.shape[0])
print(correct/testing_images.shape[0])
