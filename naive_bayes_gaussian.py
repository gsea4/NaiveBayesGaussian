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

testing_labels[testing_labels != 5] = 0
testing_labels[testing_labels == 5] = 1

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
def classify(test_set, priors, summary_5, summary_9, error_rate):
    prediction = []
    for image in test_set:
        max_pros_class = (-math.inf, -1)
        likelihood_0 = 0
        likelihood_1 = 0
        max_pros_0 = 0
        max_pros_1 = 0
        for label in range(2):
            log_prior = math.log(priors[label][1])
            p = log_prior
            # p = priors[label][1]
            for pixel in range(28 * 28):
                if label == 0:
                    mean = summary_9[pixel][0]
                    var = v0
                else:
                    mean = summary_5[pixel][0]
                    var = v1

                if pdf(image[pixel],mean,var) > 0:
                    p += math.log(pdf(image[pixel],mean,var))
                    # p *= pdf(image[pixel], mean, var)
            if label == 0:
                likelihood_0 = p
                max_pros_0 = (p, label)
            else:
                likelihood_1 = p
                max_pros_1 = (p, label)

            # if p > max_pros_class[0]:
            #     max_pros_class = (p, label)

        # t1 = likelihood_1/likelihood_0
        # t2 = priors[0][1]/priors[1][1]
        # if (likelihood_1/likelihood_0) >= (math.log(priors[0][1]))/math.log((priors[1][1])) * error_rate:
        # if (likelihood_1/likelihood_0) >= (priors[0][1]/priors[1][1]) * error_rate:
        # if (likelihood_1/likelihood_0) >= math.log(priors[0][1]/priors[1][1]) * error_rate:
        total_priors = priors[0][1] + priors[1][1]
        prior0 = priors[0][1]/total_priors
        prior1 = priors[1][1]/total_priors
        if (likelihood_1/likelihood_0) >= math.log(prior0)/math.log(prior1) * error_rate:
            max_pros_class = max_pros_1
        else:
            max_pros_class = max_pros_0

        prediction.append(max_pros_class)   
    prediction = np.array(prediction, dtype = 'int16')
    return prediction[:,1]

def calculateTPRandFPR(pred, labels):
    FP = 0
    FN = 0
    TP = 0
    TN = 0

    for i in range(testing_labels.shape[0]):
        # print("Pred is {} | Label is {}".format(pred[i], testing_labels[i]))
        if pred[i] == 1 and testing_labels[i] == 0:
            FP += 1
        elif pred[i] == 0 and testing_labels[i] == 1:
            FN += 1
        elif pred[i] == 1 and testing_labels[i] == 1:
            TP += 1
        else:
            TN += 1
        
    N_pos = TP + FN
    N_neg = TN + FP

    FPR = float(FP/N_neg)
    TPR = float(TP/N_pos)
    return TPR, FPR

def calculateAccuracy(pred, testing_labels):
    correct = 0

    for i in range(testing_labels.shape[0]):
        if pred[i] == testing_labels[i]:
            correct += 1

    print(correct)
    print(testing_images.shape[0])
    print(correct/testing_images.shape[0])
    return float(correct/testing_images.shape[0])


result1 = classify(testing_images, priors, label_5_summary, label_9_summary, 5)
TPR1, FPR1 = calculateTPRandFPR(result1, testing_labels)
calculateAccuracy(result1, testing_labels)

result2 = classify(testing_images, priors, label_5_summary, label_9_summary, 2)
TPR2, FPR2 = calculateTPRandFPR(result2, testing_labels)
calculateAccuracy(result2, testing_labels)

result3 = classify(testing_images, priors, label_5_summary, label_9_summary, 1)
TPR3, FPR3 = calculateTPRandFPR(result3, testing_labels)
calculateAccuracy(result3, testing_labels)

result4 = classify(testing_images, priors, label_5_summary, label_9_summary, 0.5)
TPR4, FPR4 = calculateTPRandFPR(result4, testing_labels)
calculateAccuracy(result4, testing_labels)

result5 = classify(testing_images, priors, label_5_summary, label_9_summary, 0.2)
TPR5, FPR5 = calculateTPRandFPR(result5, testing_labels)
calculateAccuracy(result5, testing_labels)

x = np.array([FPR1, FPR2, FPR3, FPR4, FPR5])
y = np.array([TPR1, TPR2, TPR3, TPR4, TPR5])
plt.plot(x, y)
plt.show()