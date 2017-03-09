import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten

# -----------------------------------------------------------------
# Load data
def loadData(directory):
    training_file = directory + 'train.p'
    validation_file = directory + 'valid.p'
    testing_file = directory + 'test.p'

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)

    return train, valid, test

# -----------------------------------------------------------------
# Visualize the dataSet
def inspectData(dataSet):
    # plot the number of images in each class
    plt.hist(dataSet['labels'], n_classes)

# -----------------------------------------------------------------
# Convert the image to grayscale
#
# param    img      The img to convert
# param    bPlot    True, to plot the image
# returns           The grayscaled image
#
def grayscale(img, bPlot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if (bPlot):
        plt.imshow(gray, cmap='gray')
    return gray

# -----------------------------------------------------------------
# Normalize the pixel value to (-1,1)
#
# param    data        The data to be normalized
# param    inputMin     The minimum value of the input before normalization
# param    inputMax     The maximum value of the input before normalization
# param    outputMin    The minimum value of the output after normalization
# param    outputMax    The maximum value of the output after normalization
# returns               The normalized value
#
def minMaxNormalize(data, inputMin, inputMax, outputMin, outputMax):
    return outputMin + ((data - inputMin) * (outputMax - outputMin)) / (inputMax - inputMin)


# -----------------------------------------------------------------
# Preprocess a set of images
# First grayscale
# Then normalize the pixel values between (-1, 1)
#
# param     features    The features to be preprocessed
# param     labels      The labels to be preprocessed
# returns               The list of preprocessed images
def preprocessImages(features, labels):
    # Allocate an array of the same size as the input
    # However, reduce the channels from 3 to 1
    processed_images = np.zeros(shape=(features.shape[:3]))

    # Convert all of the images to grayscale
    for idx, img in enumerate(features):
        processed_images[idx] = grayscale(img)

    # Normalize the images between (-1, 1)
    processed_images.reshape(processed_images.shape[0], processed_images.shape[1], processed_images.shape[2], 1)
    print('processed_images.shape={}, processed_images.dtype={}'.format(processed_images.shape, processed_images.dtype))
    processed_images = minMaxNormalize(processed_images, 0., 255., -1., 1.)

    processed_images, labels = shuffle(processed_images, labels)

    return processed_images, labels

# Create a 2d convolutional layer
#
# inTensor  : input tensor to this layer
# shape     : shape of the layer
# stride    : integer value that determines how far to stride each iteration
# padding   : string either 'VALID' or 'SAME'
# mean      : mean value for random variable initialization
# stddev    : standard deviation value for random variable initialization
#
def conv2d(inTensor, shape, stride, padding, mean, stddev):
    weights = tf.Variable(tf.truncated_normal(shape, mean=mean, stddev=stddev))
    bias = tf.Variable(tf.zeros(shape[3]))
    return tf.nn.conv2d(inTensor, weights, strides=[1, stride, stride, 1], padding=padding) + bias

# Create a relu layer
#
# inTensor  : input tensor to this layer
#
def relu(inTensor):
    return tf.nn.relu(inTensor)

# Create a max pooling layer
#
# inTensor  : input tensor to this layer
# ksize     : size of the pooling layer
# stride    : integer value that determines how far to stride each iteration
# padding   : string either 'VALID' or 'SAME'
#
def maxPool(inTensor, ksize, stride, padding):
    return tf.nn.max_pool(inTensor, ksize=ksize, strides=[1, stride, stride, 1], padding=padding)

# Flatten a tensor
#
# inTensor  : input tensor to this layer
#
def flat(inTensor):
    return flatten(inTensor)

# Create a fully connected layer
#
# inTensor  : input tensor to this layer
# shape     : shape of the layer
# mean      : mean value for random variable initialization
# stddev    : standard deviation value for random variable initialization
#
def fullyConnected(inTensor, shape, mean, stddev):
    weights = tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev))
    bias = tf.Variable(tf.zeros(shape[1]))
    return tf.matmul(inTensor, weights) + bias

# Given an array of layer definitions
# Construct a network and return it
#
def constructNetwork(inputTensor, layerData):
    # Construct the layers
    layer = inputTensor
    for ld in layerData:
        # Construct the layer
        constructor = ld['constructor']
        params = ld['params']
        params['inTensor'] = layer
        layer = constructor(**params)

    return layer

# -----------------------------------------------------------------
# Define a neural network
#
def DanNet():
    # Hyperparameters
    mu = 0
    sigma = 0.1

    layerData = []
    # ----------------------------------------------------------------------------------------------------------
    # Feature Extraction
    # Convolution -> MaxPool -> Convolution -> MaxPool
    #
    layerData.append({'constructor': conv2d,         'params': {'inTensor': None, 'shape': (5, 5, 3, 6), 'stride': 1, 'padding': "VALID", 'mean': mu, 'stddev': sigma}})
    layerData.append({'constructor': relu,           'params': {'inTensor': None}})
    layerData.append({'constructor': maxPool,        'params': {'inTensor': None, 'ksize': (1, 2, 2, 1), 'stride': 2, 'padding': "VALID"}})
    layerData.append({'constructor': conv2d,         'params': {'inTensor': None, 'shape': (5, 5, 6, 16),'stride': 1, 'padding': "VALID", 'mean': mu, 'stddev': sigma}})
    layerData.append({'constructor': relu,           'params': {'inTensor': None}})
    layerData.append({'constructor': maxPool,        'params': {'inTensor': None, 'ksize': (1, 2, 2, 1), 'stride': 2, 'padding': "VALID"}})
    # ----------------------------------------------------------------------------------------------------------
    # Classifier
    # Flattened Extracted Features -> Fully Connected -> Fully Connected -> Fully Connected (Classifier)
    #
    layerData.append({'constructor': flat,           'params': {'inTensor': None}})
    layerData.append({'constructor': fullyConnected, 'params': {'inTensor': None, 'shape': (400, 120), 'mean': mu, 'stddev': sigma}})
    layerData.append({'constructor': relu,           'params': {'inTensor': None}})
    layerData.append({'constructor': fullyConnected, 'params': {'inTensor': None, 'shape': (120, 84), 'mean': mu, 'stddev': sigma}})
    layerData.append({'constructor': relu,           'params': {'inTensor': None}})
    layerData.append({'constructor': fullyConnected, 'params': {'inTensor': None, 'shape': (84, n_classes), 'mean': mu, 'stddev': sigma}})

    return layerData

# -----------------------------------------------------------------
# Train a neural net
#
def trainNetwork(trainData, epochs, batchSize, learningRate):
    X_train, y_train = trainData['features'], trainData['labels']

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, n_classes)

    network = constructNetwork(x, DanNet())
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    trainer = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        print("Training...")
        print("Params: epochs:{}\tbatchSize:{}\tlearningRate:{}".format(epochs, batchSize, learningRate))
        print()
        for i in range(epochs):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, batchSize):
                end = offset + batchSize
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(trainer, feed_dict={x: batch_x, y: batch_y})

            validation_accuracy = evaluate(X_valid, y_valid, accuracy_operation, x, y)
            print("EPOCH {} ...".format(i + 1))
            print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

        saver.save(sess, './lenet')
        print("Model saved")

# --------------------------------------------------------------------------------------------------------
# Test the network
#
def testNetwork(testData):
    X_test, y_test = testData['features'], testData['labels']
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        test_accuracy = evaluate(X_test, y_test)
        print("Test Accuracy = {:.3f}".format(test_accuracy))

# --------------------------------------------------------------------------------------------------------
#
def evaluate(X_data, y_data, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# -----------------------------------------------------------------------------
# Load Data
train, valid, test = loadData('data_set/')
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

n_classes = np.max(y_train + 1)

print("Number of training examples =", X_train.shape[0])
print("Number of classes =", n_classes)

# X_train_processed, y_train = preprocessImages(X_train, y_train)
# print('max={}, min={}, mean={}, std={}'.format(np.max(X_train_processed), np.min(X_train_processed),
#                                                np.mean(X_train_processed), np.std(X_train_processed)))
#
# X_valid_processed, y_valid = preprocessImages(X_valid, y_valid)
# print('max={}, min={}, mean={}, std={}'.format(np.max(X_valid_processed), np.min(X_valid_processed),
#                                                np.mean(X_valid_processed), np.std(X_valid_processed)))

EPOCHS = np.arange(1, 20, 4)
BATCH_SIZE = 128
learningRate = 0.001

for numEpochs in EPOCHS:
    trainNetwork(train, numEpochs, BATCH_SIZE, learningRate)
