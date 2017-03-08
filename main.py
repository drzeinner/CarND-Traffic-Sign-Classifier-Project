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

# Class that defines and constructs a layer in a neural network
# Currently supports:
# conv2d
# relu
# maxPool
# fullyConnected
#
class LayerDef:

    # Data contains the following
    # layerType : string identifying what type of layer to use
    # inTensor  : input tensor to this layer
    # shape     : shape of the layer
    # mean      : mean value for random variable initialization
    # stddev    : standard deviation value for random variable initialization
    # stride    : integer value that determines how far to stride each iteration
    # padding   : string either 'VALID' or 'SAME'
    def __init__(self, data):
        if 'layerType' in data:
            self.layerType = data['layerType']
        if 'inTensor' in data:
            self.inTensor  = data['inTensor']
        if 'shape' in data:
            self.shape     = data['shape']
        if 'mean' in data:
            self.mean      = data['mean']
        if 'stddev' in data:
            self.stddev    = data['stddev']
        if 'stride' in data:
            self.stride    = data['stride']
        if 'padding' in data:
            self.padding   = data['padding']
        self.layer     = None

    def constructLayer(self):
        constructorMethod = None
        try:
            constructorMethod = getattr(self, self.layerType)
        except AttributeError:
            raise NotImplementedError(
                "Class `{}` does not implement `{}`".format(self.__class__.__name__, self.layerType))
        self.layer = constructorMethod()

    # Construct the layer based its settings
    def conv2d(self):
        weights = tf.Variable(tf.truncated_normal(self.shape, mean=self.mean, stddev=self.stddev))
        bias = tf.Variable(tf.zeros(self.shape[3]))
        return tf.nn.conv2d(self.inTensor, weights, strides=[1, self.stride, self.stride, 1], padding=self.padding) + bias

    # Construct the layer based its settings
    def relu(self):
        return tf.nn.relu(self.inTensor)

    # Construct the layer based its settings
    def maxPool(self):
        return tf.nn.max_pool(self.inTensor, ksize=self.shape, strides=[1, self.stride, self.stride, 1], padding=self.padding)

    # Construct the layer based its settings
    def fullyConnected(self):
        weights = tf.Variable(tf.truncated_normal(shape=self.shape, mean=self.mean, stddev=self.stddev))
        bias = tf.Variable(self.shape[1])
        return tf.matmul(self.inTensor, weights) + bias

# -----------------------------------------------------------------
def DanNet(x):
    # Hyperparameters
    mu = 0
    sigma = 0.1


    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_data = {
        'layerType': "conv2d",
        'inTensor' : x,
        'shape'    : (5, 5, 3, 6),
        'mean'     : mu,
        'stddev'    : sigma,
        'stride'   : 1,
        'padding'  : "VALID"
    }
    conv1Def = LayerDef(conv1_data)
    conv1Def.constructLayer()

    # SOLUTION: Activation.
    conv1_activation_data = {
        'layerType'     : "relu",
        'inTensor' : conv1Def.layer
    }
    conv1ActivationDef = LayerDef(conv1_activation_data)
    conv1ActivationDef.constructLayer()

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1_maxpool_data = {
        'layerType'     : "maxPool",
        'inTensor' : conv1ActivationDef.layer,
        'shape'    : (1, 2, 2, 1),
        'stride'  : 2,
        'padding'  : "VALID"
    }
    conv1MaxPoolDef = LayerDef(conv1_maxpool_data)
    conv1MaxPoolDef.constructLayer()

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1MaxPoolDef.layer, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

# --------------------------------------------------------------------------------------------------------
#
def evaluate(X_data, y_data):
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
image_shape = [X_train.shape[1], X_train.shape[2]]

print("Number of training examples =", X_train.shape[0])
print("Number of testing examples =", X_test.shape[0])
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

# X_train_processed, y_train = preprocessImages(X_train, y_train)
# print('max={}, min={}, mean={}, std={}'.format(np.max(X_train_processed), np.min(X_train_processed),
#                                                np.mean(X_train_processed), np.std(X_train_processed)))
#
# X_valid_processed, y_valid = preprocessImages(X_valid, y_valid)
# print('max={}, min={}, mean={}, std={}'.format(np.max(X_valid_processed), np.min(X_valid_processed),
#                                                np.mean(X_valid_processed), np.std(X_valid_processed)))

EPOCHS = 10
BATCH_SIZE = 128
learningRate = 0.001

# Set up
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)


logits = DanNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
training_operation = optimizer.minimize(loss_operation)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))

    saver.save(sess, './lenet')
    print("Model saved")

# with tf.Session() as sess:
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {:.3f}".format(test_accuracy))