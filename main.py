import pickle
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.contrib.layers import flatten
from math import ceil
from glob import glob

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

    print('train size:{}, valid size:{}, test size:{}'.format(len(train['features']), len(valid['features']),  len(test['features'])))

    return train, valid, test

# ---------------------------------------------------------------------
# Given train, validation and test datasets
# Combine them into one set
# Shuffle them randomly
# And split back into train, validation and test sets
#
def combineAndSplit(train, valid, test):
    # Combine
    totalFeatures = np.concatenate([train['features'], valid['features'], test['features']])
    totalLabels = np.concatenate([train['labels'], valid['labels'], test['labels']])

    # Shuffle
    randFeatures, randLabels = shuffle(totalFeatures, totalLabels)

    # Split

    # test 60%
    dataSetSize = len(randFeatures)
    print("dataSetSize={}".format(dataSetSize))
    trainStartIdx = 0
    trainEndIdx = round(dataSetSize * .6)
    newTrainFeatures = randFeatures[trainStartIdx:trainEndIdx]
    newTrainLabels = randLabels[trainStartIdx:trainEndIdx]
    newTrain = {'features': newTrainFeatures, 'labels': newTrainLabels}
    print('trainStart:{} trainEnd:{} trainSize={}'.format(trainStartIdx, trainEndIdx, len(newTrainFeatures)))

    # Valid 20%
    validStartIdx = trainEndIdx + 1
    validEndIdx = validStartIdx + round(dataSetSize * .2)
    newValidFeatures = randFeatures[validStartIdx:validEndIdx]
    newValidLabels = randLabels[validStartIdx:validEndIdx]
    newValid = {'features': newValidFeatures, 'labels': newValidLabels}
    print('validStart:{} validEnd:{} validSize={}'.format(validStartIdx, validEndIdx, len(newValidFeatures)))

    # Test 20%
    testStartIdx = validEndIdx + 1
    testEndIdx = dataSetSize - 1
    newTestFeatures = randFeatures[testStartIdx:testEndIdx]
    newTestLabels = randLabels[testStartIdx:testEndIdx]
    newTest = {'features': newTestFeatures, 'labels': newTestLabels}
    print('testStart:{} testEnd:{} testSize={}'.format(testStartIdx, testEndIdx, len(newTestFeatures)))

    return newTrain, newValid, newTest




# =====================================================================================================
# DATA INSPECTION
# =====================================================================================================

# -----------------------------------------------------------------
# Display a grid of images
#
# param     imageSet        The set of all images to pull from
# param     indices         The indices of the images to display
# param     fileName        Name of the image to save
# param     numCols         The number of columns in the grid
# param     bNormalized     True if the imageset is normalized to (-1,1)
#
def displayImageGrid(imageSet, indices, fileName, numCols=10, bNormalized=True):
    numImages = len(indices)
    numRows = ceil(numImages/numCols)
    rows = []
    for i in range(numRows):
        imageRow = []
        for j in range(numCols):
            idx = i*numCols + j
            if (idx >= numImages):
                imageRow.append(np.zeros([32,32,1]))
            else:
                image = imageSet[indices[idx]]
                if bNormalized:
                    image = minMaxNormalize(image, -1., 1., 0., 255.)
                imageRow.append(image)
        rows.append(np.hstack(imageRow))

    output = np.vstack(rows)
    cv2.imwrite('visualization/' + fileName, output)

# -----------------------------------------------------------------
# Visualize the dataSet
#
# param     bDataDist               Create a histogram of the data set distribution
# param     bRandomImage            Display a randomImage
# param     bCompareProcessing      Compare the source image to the processed images
#
def inspectData(bDataDist=False, bRandomImage=False, bCompareProcessing=False):
    train, valid, test = loadData('data_set/')
    if bDataDist:
        # plot the number of images in each class
        plt.hist(train['labels'], 43, alpha=0.5, label='train')
        plt.hist(valid['labels'], 43, alpha=0.5, label='valid')
        plt.savefig('visualization/histogram.png')
    if bRandomImage:
        randomImage = train['features'][np.random.randint(0, train['features'].shape[0])]
        plt.imsave('visualization/randomImage.png', randomImage)
    if bCompareProcessing:
        pretrain, pvalid, ptest = loadData('preprocess/')
        numCols = 10
        numRows = 4
        orig_rows = []
        proc_rows = []
        for i in range(numRows):
            startIndex = np.random.randint(0, pretrain['features'].shape[0])
            orig_rows.append(np.hstack(train['features'][startIndex:startIndex+numCols]))

            pImages = minMaxNormalize(pretrain['features'][startIndex:startIndex+numCols], -1., 1., 0., 255.)
            # print('max={}, min={}, mean={}, std={}'.format(np.max(pImages), np.min(pImages),
            #                                                np.mean(pImages), np.std(pImages)))
            proc_rows.append(np.hstack(pImages))
            # print("{} startIndex={}".format(i, startIndex))
            print("labels:{}".format(pretrain['labels'][startIndex:startIndex+numCols]))

        orig = np.vstack(orig_rows)
        orig_color = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        proc = np.vstack(proc_rows)
        cv2.imwrite('visualization/orig.png', orig_color)
        cv2.imwrite('visualization/proc.png', proc)


# =====================================================================================================
# DATA PREPROCESSING
# =====================================================================================================

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


# -----------------------------------------------------------------
# Preprocess a single image and return the result
#
def preprocessImage(image):
    # make sure the image is 32x32
    scaleFactorY = 32. / image.shape[0]
    scaleFactorX = 32. / image.shape[1]
    resized = cv2.resize(image, None, fx=scaleFactorX, fy=scaleFactorY, interpolation=cv2.INTER_AREA)
    gray = grayscale(resized)
    equ = cv2.equalizeHist(gray).reshape(32,32,1)
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
        feature = brighten(feature, .5)
        features[i] = feature
    return features

# ---------------------------------------------------------------------
# Preprocess the original data set
#
def preprocessData():
    ptrain, pvalid, ptest = loadData('data_set/')
    ntrain, nvalid, ntest = combineAndSplit(ptrain, pvalid, ptest)
    # ptrain['features'] = augmentData(ptrain['features'])
    preprocessImages(ntrain, 'train.p')
    preprocessImages(nvalid, 'valid.p')
    preprocessImages(ntest, 'test.p')


# =====================================================================================================
# NEURAL NETWORK
# =====================================================================================================


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

def constructNetwork(x, y, keep_prob, regScalar, learningRate=0.001):
    numClasses = 43
    # ----------------------------------------------------------------------------------
    # Construct the network
    # network = constructNetwork(x, DanNet())
    one_hot_y = tf.one_hot(y, numClasses)
    mu = 0
    sigma = 0.01
    # seed the rng to reduce the variation while testing different hyperparameters
    # tf.set_random_seed(1234)

    weights = {'conv1': tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 16), mean=mu, stddev=sigma)),
               'conv2': tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 32), mean=mu, stddev=sigma)),
               'conv3': tf.Variable(tf.truncated_normal(shape=(1, 1, 32, 16), mean=mu, stddev=sigma)),

               'fc1': tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma)),
               'fc2': tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma)),
               'fc3': tf.Variable(tf.truncated_normal(shape=(84, numClasses), mean=mu, stddev=sigma))}
    biases = {'conv1': tf.Variable(tf.zeros(16)),
              'conv2': tf.Variable(tf.zeros(32)),
              'conv3': tf.Variable(tf.zeros(16)),

              'fc1': tf.Variable(tf.zeros(120)),
              'fc2': tf.Variable(tf.zeros(84)),
              'fc3': tf.Variable(tf.zeros(numClasses))}
    # ----------------------------------------------------------------------------------------------------------
    # Feature Extraction
    # Convolution -> MaxPool -> Convolution -> MaxPool
    #
    network = conv2d(x, weights['conv1'], biases['conv1'], 1, 'VALID', 'conv1')  # 28x28x16
    network = relu(network, 'activate1')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')  # 14x14x16
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv2'], biases['conv2'], 1, 'VALID', 'conv2')  # 10x10x32
    network = relu(network, 'activate2')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')  # 5x5x32
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv3'], biases['conv3'], 1, 'VALID', 'conv3')  # 5x5x16
    network = relu(network, 'activate3')

    network = conv2d(x, weights['conv1'], biases['conv1'], 1, 'VALID', 'conv1')  # 28x28x16
    network = relu(network, 'activate1')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')  # 14x14x16
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv2'], biases['conv2'], 1, 'VALID', 'conv2')  # 10x10x32
    network = relu(network, 'activate2')
    network = maxPool(network, (1, 2, 2, 1), 2, 'VALID')  # 5x5x32
    network = dropout(network, keep_prob)

    network = conv2d(network, weights['conv3'], biases['conv3'], 1, 'VALID', 'conv3')  # 5x5x16
    network = relu(network, 'activate3')

    # network = conv2d(network, weights['conv3'], biases['conv3'], 1, 'VALID')    # 5x5x16
    # network = dropout(network, keep_prob)

    # ----------------------------------------------------------------------------------------------------------
    # Classifier
    # Flattened Extracted Features -> Fully Connected -> Fully Connected -> Fully Connected (Classifier)
    #
    network = flat(network)  # 400x1
    network = fullyConnected(network, weights['fc1'], biases['fc1'])  # 400x300
    network = relu(network, 'activatefc1')
    # network = fullyConnected(network, weights['fc4'], biases['fc4'])           # 300x120
    # network = relu(network)
    network = fullyConnected(network, weights['fc2'], biases['fc2'])  # 120x84
    network = relu(network, 'activatefc2')
    network = dropout(network, keep_prob)
    network = fullyConnected(network, weights['fc3'], biases['fc3'])  # 84x43
    # ----------------------------------------------------------------------------------

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=one_hot_y, name='cross_entropy')

    # Apply l2 regularization to the weights
    # to help prevent overfitting
    regularization_term = 0
    for key, value in weights.items():
        regularization_term = regScalar * tf.nn.l2_loss(value)

    loss_operation = tf.reduce_mean(cross_entropy + regularization_term)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate)
    trainer = optimizer.minimize(loss_operation, name='trainer')

    prediction = tf.argmax(network, 1, name='prediction')
    correct_answer = tf.argmax(one_hot_y, 1, name='correct_answer')
    correct_prediction = tf.equal(prediction, correct_answer, name='correct_prediction')
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy_operation')

    return network

# -----------------------------------------------------------------
# Train a neural net
#
def trainNetwork(trainData, validationData, testData, epochs, batchSize, learningRate):
    X_train, y_train = trainData['features'], trainData['labels']
    X_valid, y_valid = validationData['features'], validationData['labels']
    X_test, y_test = testData['features'], testData['labels']
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    regScalar = tf.placeholder(tf.float32)

    constructNetwork(x, y, keep_prob, regScalar, learningRate)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        trainer = sess.graph.get_operation_by_name('trainer')
        accuracy_operation = sess.graph.get_tensor_by_name('accuracy_operation:0')

        num_examples = len(X_train)
        print("Training...")
        print("Params: epochs:{}\tbatchSize:{}\tlearningRate:{}".format(epochs, batchSize, learningRate))
        print()
        train_plot = []
        valid_plot = []
        train_accuracy = 0.
        validation_accuracy = 0.
        test_accuracy = 0.
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
            test_accuracy = evaluate(X_test, y_test, accuracy_operation, x, y, keep_prob, regScalar, batchSize)
            print("EPOCH {} ...".format(i + 1))
            print("Accuracy Training = {:.3f}  Validation = {:.3f} Test = {:.3f}\n".format(train_accuracy, validation_accuracy, test_accuracy))

        # Generate the learning curve plot
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
        pltFileName = 'visualization/plots/Accuracy_{}_{}_{}.png'.format(idx, idx2, idx3)
        plt.savefig(pltFileName)
        print("Plot saved in {}".format(pltFileName))

        # Save the model
        saver.save(sess, 'models/lenet')
        print("Model saved")

# -----------------------------------------------------------------------------------------------
# Train the network
# potentially multiple times with different hyperparameters
#
def train():
    global num_classes
    global idx
    global idx2
    global idx3

    # -----------------------------------------------------------------------------
    # Load Data
    trainData, validData, testData = loadData('preprocess/')
    for idx, numEpochs in enumerate(EPOCHS):
        for idx2, batchSize in enumerate(BATCH_SIZE):
            for idx3, learnRate in enumerate(LEARN_RATES):
                trainNetwork(trainData, validData, testData, numEpochs, batchSize, learnRate)


# =====================================================================================================
# NETWORK EVALUATION
# =====================================================================================================

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

# ------------------------------------------------------------------------------------------
# Outputs the feature maps for a particular image
#
# param     image           The image to get feature maps for
# param     featureMaps     Array of feature maps to get
#
def outputFeatureMap(image, featureMaps, activation_min=-1, activation_max=-1, plt_num=1):
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    regScalar = tf.placeholder(tf.float32)

    constructNetwork(x, y, keep_prob, regScalar)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'models/lenet')
        plt.imsave('visualization/featureMaps/featureMapOrig.png', image.reshape(32, 32))
        for featureName in featureMaps:
            tf_activation = sess.graph.get_tensor_by_name(featureName)
            activation = tf_activation.eval(session=sess, feed_dict={x: [image], keep_prob: dropOutProb,
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
            featureMapsFileName = 'visualization/featureMaps/' + featureName + '.png'
            plt.savefig(featureMapsFileName)
            print("FeatureMaps saved in {}".format(featureMapsFileName))


# -----------------------------------------------------------------
# Evaluate the missed classifications
#
def evaluateMisses(testData):
    features, labels = testData['features'], testData['labels']
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    regScalar = tf.placeholder(tf.float32)

    constructNetwork(x, y, keep_prob, regScalar)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'models/lenet')

        prediction = sess.graph.get_tensor_by_name('prediction:0')
        correct_answer = sess.graph.get_tensor_by_name('correct_answer:0')
        correct_prediction = sess.graph.get_tensor_by_name('correct_prediction:0')
        pred = sess.run(prediction, feed_dict={x: features, keep_prob: 1.0, regScalar: 0.0})
        answer = sess.run(correct_answer, feed_dict={y: labels, keep_prob: 1.0, regScalar: 0.0})
        correct = sess.run(correct_prediction, feed_dict={x:features, y:labels, keep_prob:1.0, regScalar:0})
        incorrectIndices = np.where(correct == False)[0]
        # print("Prediction:{} Answer:{}".format(pred, answer))
        print("Num Incorrect:{}/{}".format(incorrectIndices.shape, features.shape[0]))
        print("predictions:{}".format(pred[incorrectIndices]))
        print("=====================================================")
        print("answers:{}".format(answer[incorrectIndices]))
        print("=====================================================")
        print("indices:{}".format(incorrectIndices))

        displayImageGrid(features, incorrectIndices, 'incorrectPredictions.png', numCols=12)


# =====================================================================================================
# PREDICTION
# =====================================================================================================
# ----------------------------------------------------------------
# Load and return the test images
#
def loadTestImages():
    fileNames = glob('testImages/*.png')
    fileNames.sort()

    images = []
    for fileName in fileNames:
        imageName = fileName[:-3]
        img = cv2.imread(fileName)
        images.append(img)
    return images

# ----------------------------------------------------------------
# Given a set of images
# Classify them
#
def classifyImages(images, labels):

    # Preprocess the images
    processedImages = []
    for image in images:
        processedImages.append(preprocessImage(image))

    displayImageGrid(processedImages, np.arange(0, len(images)), "testImages.png", len(images))

    # Convert the labels into one-hot format
    # one_hot_labels = np.zeros((len(labels), num_classes))
    # for i, label in enumerate(labels):
    #     one_hot_labels[i][label] = 1

    # Now Classify them
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    y = tf.placeholder(tf.int32, (None))
    keep_prob = tf.placeholder(tf.float32)
    regScalar = tf.placeholder(tf.float32)

    logits = constructNetwork(x, y, keep_prob, regScalar)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, 'models/lenet')

        prediction = sess.graph.get_tensor_by_name('prediction:0')
        correct_answer = sess.graph.get_tensor_by_name('correct_answer:0')
        correct_prediction = sess.graph.get_tensor_by_name('correct_prediction:0')
        accuracy_operation = sess.graph.get_tensor_by_name('accuracy_operation:0')

        pred = sess.run(prediction, feed_dict={x: processedImages, keep_prob: 1.0, regScalar: 0.0})
        print("Predictions:{}".format(pred))
        answer = sess.run(correct_answer, feed_dict={y: labels, keep_prob: 1.0, regScalar: 0.0})
        print("Answers:{}".format(answer))
        correct = sess.run(correct_prediction,
                           feed_dict={x: processedImages, y: labels, keep_prob: 1.0, regScalar: 0})
        print("Correct:{}".format(correct))
        accuracy = sess.run(accuracy_operation,
                            feed_dict={x: processedImages, y: labels, keep_prob: 1.0, regScalar: 0.0})
        print("Accuracy: {:.0f}%".format(accuracy * 100.))
        softmax = sess.run(tf.nn.top_k(tf.nn.softmax(logits), k=5),
                           feed_dict={x: processedImages, y: labels, keep_prob: 1.0, regScalar: 0.0})
        print("Softmax: {}".format(softmax))

# -----------------------------------------------------------------
# Globals
idx = 0
idx2 = 0
idx3 = 0
num_classes = 43
dropOutProb = 0.5
regScalarValue = 0.01
EPOCHS = [60]  # np.arange(2, 20, 4)
BATCH_SIZE = [128]  # , 256, 512, 1024, 2048]
LEARN_RATES = [0.001]  # , 0.01, 0.1, 1.0]

# -----------------------------------------------------------------------------
# Execute code here


# inspectData(bDataDist=False, bRandomImage=False, bCompareProcessing=True)
# preprocessData()
# train()
classifyImages(loadTestImages(), [29, 28, 38, 17,13])
# trainData, validData, testData = loadData('preprocess/')
# evaluateMisses(validData)
# outputFeatureMap(trainData['features'][100], featureMaps=['activate1:0', 'activate2:0', 'activate3:0'])

