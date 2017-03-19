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
def inspectData():
    train, valid, test = loadData('data_set/')
    pretrain, pvalid, ptest = loadData('preprocess/')
    # plot the number of images in each class
    # plt.hist(train['labels'], 43, alpha=0.5, label='train')
    # plt.hist(valid['labels'], 43, alpha=0.5, label='valid')
    # displayRandomImage(pretrain['features'])

    numCols = 10
    numRows = 4
    orig_rows = []
    proc_rows = []
    for i in range(numRows):
        startIndex = np.random.randint(0, pretrain['features'].shape[0])
        orig_rows.append(np.hstack(train['features'][startIndex:startIndex+numCols]))

        pImages = minMaxNormalize(pretrain['features'][startIndex:startIndex+numCols], -1., 1., 0., 255.)
        print('max={}, min={}, mean={}, std={}'.format(np.max(pImages), np.min(pImages),
                                                       np.mean(pImages), np.std(pImages)))
        proc_rows.append(np.hstack(pImages))
        print("{} startIndex={}".format(i, startIndex))

    orig = np.vstack(orig_rows)
    proc = np.vstack(proc_rows)
    cv2.imwrite('orig.png', orig)
    cv2.imwrite('proc.png', proc)

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
# Apply Canny Edge Transform
#
# param    img              The img to convert
# param    low_threshold    Low value threshold to include
# returns  high_threshold   High value threshold to include
#
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

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


def preprocessImage(image):
    gray = grayscale(image)
    equ = cv2.equalizeHist(gray).reshape(32, 32, 1)
    normalized = minMaxNormalize(equ, 0., 255., -1., 1.)
    return normalized

# -----------------------------------------------------------------
# Preprocess a set of images
# First grayscale
# apply histogram equalization
# Then normalize the pixel values between (-1, 1)
#
# param     data            The data to be preprocessed
# param     outfileName     The name of the file to save the processed data to
# return                    The processed data
def preprocessImages(data, outfileName=''):
    X, y = data['features'], data['labels']
    newX = np.empty([X.shape[0], X.shape[1], X.shape[2], 1])

    # Convert all of the images to grayscale
    for i, img in enumerate(X):
        gray = grayscale(img)
        equ = cv2.equalizeHist(gray).reshape(32,32,1)
        newX[i] = equ

    # Normalize the images between (-1, 1)
    newX = minMaxNormalize(newX, 0., 255., -1., 1.)

    newData = {'features': newX, 'labels': y}

    # Save the processed data to disk
    if (len(outfileName) > 0):
        pickle.dump(newData, open("preprocess/" + outfileName, "wb"))

    return newData

# Rotate an image randomly
#
# param    img            The image to rotate
# param    maxRotation    The maximum allowed rotation
# returns                 The rotated image
def rotateRandom(img, maxRotation):
    rows = img.shape[0]
    cols = img.shape[1]
    rotation = np.random.rand() * 2 * maxRotation - maxRotation
    # print("rotation={}".format(rotation))
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    return dst

# Add a random amount of brightness to the image
#
# param     img             The image to brighten
# param     maxBrightness   The maxmimum amount of brightness to apply
# returns                   The brightened image
#
def brighten(img, maxBrightness):
    tempImg = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    tempImg.astype(dtype=np.float64)
    brightnessValue = np.random.rand() * maxBrightness
    tempImg[:,:,2] = tempImg[:,:,2]*brightnessValue
    # Clamp to no bigger than 255
    tempImg[:,:,2][tempImg[:,:,2]>255] = 255
    tempImg.astype(dtype=np.uint8)
    return cv2.cvtColor(tempImg,cv2.COLOR_HSV2RGB)

# Augment the training set to add more variation
#
# param     features      The training set of features to be augmented
# returns                 The augmented set
#
def augmentData(features):
    for i, feature in enumerate(features):
        feature = rotateRandom(feature, 30)
        #feature = brighten(feature, .5)
        # TODO : do other augmentation
        features[i] = feature
    return features


# Create a 2d convolutional layer
#
# inTensor  : input tensor to this layer
# shape     : shape of the layer
# stride    : integer value that determines how far to stride each iteration
# padding   : string either 'VALID' or 'SAME'
# mean      : mean value for random variable initialization
# stddev    : standard deviation value for random variable initialization
# name      : name of the layer
#
def conv2d(inTensor, weights, bias, stride, padding, name):
    return tf.nn.conv2d(inTensor, weights, strides=[1, stride, stride, 1], padding=padding, name=name) + bias

# Create a relu layer
#
# inTensor  : input tensor to this layer
#
def relu(inTensor, name):
    return tf.nn.relu(inTensor, name=name)

# Create a tanh layer
#
# inTensor  : input tensor to this layer
#
def tanh(inTensor):
    return tf.nn.tanh(inTensor)

# Create a sigmoid layer
#
# inTensor  : input tensor to this layer
#
def sigmoid(inTensor):
    return tf.nn.sigmoid(inTensor)

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
def fullyConnected(inTensor, weights, bias):
    return tf.matmul(inTensor, weights) + bias

# Create a dropout layer
#
# inTensor  : input tensor to this layer
# dropout   : value to dropout
#
def dropout(inTensor, dropout):
    return tf.nn.dropout(inTensor, dropout)

# -----------------------------------------------------------------
# Train a neural net
#
def trainNetwork(trainData, validationData, testData, epochs, batchSize, learningRate, bRestore, featureMaps=[]):
    X_train, y_train = trainData['features'], trainData['labels']
    X_valid, y_valid = validationData['features'], validationData['labels']
    X_test, y_test = testData['features'], testData['labels']

    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, num_classes)
    keep_prob = tf.placeholder(tf.float32)
    regScalar = tf.placeholder(tf.float32)

    # ----------------------------------------------------------------------------------
    # Construct the network
    # network = constructNetwork(x, DanNet())

    mu = 0
    sigma = 0.01
    # TODO : remove seeding for final model
    tf.set_random_seed(1234)

    weights = {'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean=mu, stddev=sigma)),
               'conv2': tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma)),
               'conv3': tf.Variable(tf.truncated_normal(shape=(1, 1, 32, 16), mean=mu, stddev=sigma)),
               'fc1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
               # 'fc4': tf.Variable(tf.truncated_normal(shape=(250, 120), mean=mu, stddev=sigma)),
               'fc2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
               'fc3': tf.Variable(tf.truncated_normal(shape=(84, num_classes), mean=mu, stddev=sigma))}
    biases = {'conv1': tf.Variable(tf.zeros(16)),
              'conv2': tf.Variable(tf.zeros(32)),
              'conv3': tf.Variable(tf.zeros(16)),
              'fc1': tf.Variable(tf.zeros(120)),
              # 'fc4': tf.Variable(tf.zeros(120)),
              'fc2': tf.Variable(tf.zeros(84)),
              'fc3': tf.Variable(tf.zeros(num_classes))}
    # ----------------------------------------------------------------------------------------------------------
    # Feature Extraction
    # Convolution -> MaxPool -> Convolution -> MaxPool
    #
    network = conv2d(x, weights['conv1'], biases['conv1'], 1, 'VALID', 'conv1')             # 28x28x16
    network = relu(network, 'activate1')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')                                    # 14x14x16
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv2'], biases['conv2'], 1, 'VALID', 'conv2')       # 10x10x32
    network = relu(network, 'activate2')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')                                    # 5x5x32
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv3'], biases['conv3'], 1, 'VALID', 'conv3')       # 5x5x16
    network = relu(network, 'activate3')

    #network = conv2d(network, weights['conv3'], biases['conv3'], 1, 'VALID')    # 5x5x16
    #network = dropout(network, keep_prob)

    # ----------------------------------------------------------------------------------------------------------
    # Classifier
    # Flattened Extracted Features -> Fully Connected -> Fully Connected -> Fully Connected (Classifier)
    #
    network = flat(network)                                                     # 400x1
    network = fullyConnected(network, weights['fc1'], biases['fc1'])            # 400x300
    network = relu(network, 'activatefc1')
    # network = fullyConnected(network, weights['fc4'], biases['fc4'])           # 300x120
    # network = relu(network)
    network = fullyConnected(network, weights['fc2'], biases['fc2'])            # 120x84
    network = relu(network, 'activatefc2')
    network = dropout(network, keep_prob)
    network = fullyConnected(network, weights['fc3'], biases['fc3'])            # 84x43
    # ----------------------------------------------------------------------------------

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=one_hot_y)

    # Apply l2 regularization to the weights
    # to help prevent overfitting
    regularization_term = 0
    for key, value in weights.items():
        regularization_term = regScalar * tf.nn.l2_loss(value)

    loss_operation = tf.reduce_mean(cross_entropy + regularization_term)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    trainer = optimizer.minimize(loss_operation)

    prediction = tf.argmax(network, 1)
    correct_answer = tf.argmax(one_hot_y, 1)
    correct_prediction = tf.equal(prediction, correct_answer)
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if (bRestore):
            saver.restore(sess, './lenet')
            train_accuracy = evaluate(X_train, y_train, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
            validation_accuracy = evaluate(X_valid, y_valid, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
            print("Accuracy Training = {:.3f}  Validation = {:.3f}\n".format(train_accuracy, validation_accuracy))

            # test_accuracy = evaluate(X_test, y_test, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
            # print("Accuracy Test = {:.3f}\n".format(test_accuracy))

            randomImage = X_train[np.random.randint(0, X_train.shape[0])]
            outputFeatureMap(sess, randomImage, featureMaps, x, keep_prob, regScalar)
        else:
            sess.run(tf.global_variables_initializer())

            num_examples = len(X_train)

            print("Training...")
            print("Params: epochs:{}\tbatchSize:{}\tlearningRate:{}".format(epochs, batchSize, learningRate))
            print("X_train.shape={}".format(X_train.shape))
            print()
            train_plot = []
            valid_plot = []
            train_accuracy = 0.
            validation_accuracy = 0.
            for i in range(epochs):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, batchSize):
                    end = offset + batchSize
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(trainer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropOutProb, regScalar: regScalarValue})

                # Calculate accuracy at this point
                # on the training set
                # and the validation set
                train_accuracy = evaluate(X_train, y_train, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
                train_plot.append(train_accuracy)
                validation_accuracy = evaluate(X_valid, y_valid, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
                valid_plot.append(validation_accuracy)
                print("EPOCH {} ...".format(i + 1))
                print("Accuracy Training = {:.3f}  Validation = {:.3f}\n".format(train_accuracy, validation_accuracy))

            # Calculate the top 5 classes that were predicted incorrectly

            # Calculate the confusion matrix
            # true_pos = tf.equal(tf.argmax(network, 1), tf.argmax(one_hot_y, 1))
            # true_neg = tf.equal(tf.argmax(network, 0), tf.argmax(one_hot_y, 1))
            # false_pos = tf.equal(tf.argmax(network, 1), tf.argmax(one_hot_y, 0))
            # false_neg = tf.equal(tf.argmax(network, 0), tf.argmax(one_hot_y, 0))

            # Generate the learning curve plot
            # y_axis_len = len(valid_plot)
            x_axis = np.arange(0, epochs, 1)
            fig, ax = plt.subplots(1)
            ax.plot(x_axis, train_plot, lw=2, label='train', color='blue')
            ax.plot(x_axis, valid_plot, lw=1, label='valid', color='red', ls='--')
            ax.legend(loc='upper left')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('Accuracy')
            plt.title("Params: epochs:{}    batchSize:{}    learningRate:{}".format(epochs, batchSize, learningRate))
            ax.text(0.95, 0.01, "train:{:.3f}   valid:{:.3f}".format(train_accuracy, validation_accuracy),
                    verticalalignment='bottom', horizontalalignment='right',
                    transform=ax.transAxes,
                    color='green', fontsize=15)
            pltFileName = 'Plots/Accuracy_{}_{}_{}.png'.format(idx, idx2, idx3)
            plt.savefig(pltFileName)
            print("Plot saved in {}".format(pltFileName))

            # Save the model
            saver.save(sess, './lenet')
            print("Model saved")

# --------------------------------------------------------------------------------------------------------
# Calculate the accuracy of the model
#
def evaluate(X_data, y_data, accuracy_operation, x, y, keep_prob, regScalar, batchSize):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batchSize):
        batch_x, batch_y = X_data[offset:offset+batchSize], y_data[offset:offset+batchSize]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, regScalar: 0.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# --------------------------------------------------------------------------------------------------------
# Make a prediction on an image
#
# def predict(img, sess):


# --------------------------------------------------------------------------------------------------------
# Calculate the confusion matrix of the model
#
# def confusionMatrix(X_data, y_data, operations, x, y, keep_prob, regScalar, batchSize):
#     num_examples = len(X_data)
#     total_accuracy = 0
#     sess = tf.get_default_session()
#     for offset in range(0, num_examples, batchSize):
#         batch_x, batch_y = X_data[offset:offset+batchSize], y_data[offset:offset+batchSize]
#         accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0, regScalar: 0.0})
#         total_accuracy += (accuracy * len(batch_x))
#     return total_accuracy / num_examples

# -----------------------------------------------------------------
# Visualize the dataSet
def displayRandomImage(features):
    randomImage = features[np.random.randint(0, features.shape[0])]
    # plt.imshow(randomImage)
    print('max={}, min={}, mean={}, std={}'.format(np.max(randomImage), np.min(randomImage),
                                                   np.mean(randomImage), np.std(randomImage)))
    plt.imsave('randomImage.png', randomImage)

def preprocessData():
    ptrain, pvalid, ptest = loadData('data_set/')
    # ptrain['features'] = augmentData(ptrain['features'])
    preprocessImages(ptrain, 'train.p')
    preprocessImages(pvalid, 'valid.p')
    preprocessImages(ptest, 'test.p')

def train(bRestore, featureMaps=[]):
    global num_classes
    global idx
    global idx2
    global idx3

    # -----------------------------------------------------------------------------
    # Load Data
    train, valid, test = loadData('preprocess/')

    print('max={}, min={}, mean={}, std={}'.format(np.max(train['features']), np.min(train['features']),
                                                   np.mean(train['features']), np.std(train['features'])))
    num_classes = np.max(train['labels'] + 1)
    for idx, numEpochs in enumerate(EPOCHS):
        for idx2, batchSize in enumerate(BATCH_SIZE):
            for idx3, learnRate in enumerate(LEARN_RATES):
                trainNetwork(train, valid, test, numEpochs, batchSize, learnRate, bRestore, featureMaps)

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(sess, image_input, featureMaps, x, keep_prob, regScalar, activation_min=-1, activation_max=-1, plt_num=1):
    activation_min = 1
    activation_max = 1
    plt_num = 1
    processedImage = [image_input]
    plt.imsave('featureMaps/featureMapOrig.png', image_input.reshape(32, 32))
    for featureName in featureMaps:
        tf_activation = sess.graph.get_tensor_by_name(featureName)
        # outputFeatureMap(image_input=[randomImage], tf_activation=layer, x=x)
        activation = tf_activation.eval(session=sess, feed_dict={x: processedImage, keep_prob: dropOutProb,
                                                                 regScalar: regScalarValue})
        featuremaps = activation.shape[-1]
        plt.figure(plt_num, figsize=(15, 15))
        for featuremap in range(featuremaps):
            plt.subplot(6, 8, featuremap + 1)  # sets the number of feature maps to show on each row and column
            plt.title('' + str(featuremap))  # displays the feature map number
            if activation_min != -1 & activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                           vmax=activation_max, cmap="gray")
            elif activation_max != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmax=activation_max,
                           cmap="gray")
            elif activation_min != -1:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", vmin=activation_min,
                           cmap="gray")
            else:
                plt.imshow(activation[0, :, :, featuremap], interpolation="nearest", cmap="gray")
        featureMapsFileName = 'featureMaps/' + featureName + '.png'
        plt.savefig(featureMapsFileName)
        print("FeatureMaps saved in {}".format(featureMapsFileName))

# -----------------------------------------------------------------
# Globals
idx = 0
idx2 = 0
idx3 = 0
num_classes = 0
dropOutProb = 0.5
regScalarValue = 0.01
EPOCHS = [120]  # np.arange(2, 20, 4)
BATCH_SIZE = [128]  # , 256, 512, 1024, 2048]
LEARN_RATES = [0.001]  # , 0.01, 0.1, 1.0]

# -----------------------------------------------------------------------------
# Execute code here

bTrain = False

if bTrain:
    preprocessData()
    inspectData()
    train(False)
else:
    train(True, ['activate1:0', 'activate2:0', 'activate3:0'])
