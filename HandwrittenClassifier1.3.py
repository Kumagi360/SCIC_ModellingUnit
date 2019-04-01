import numpy as np

import tensorflow as tf

from PIL import Image

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from tensorflow.examples.tutorials.mnist import input_data


accuracyDict = {}


def runVarNodes(h1_nodes, h2_nodes, h3_nodes):

    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    tf.logging.set_verbosity(old_v)

    n_train = mnist.train.num_examples # 55,000
    n_validation = mnist.validation.num_examples # 5000
    n_test = mnist.test.num_examples # 10,000

    n_input = 784   # input layer (28x28 pixels)
    n_hidden1 = int(h1_nodes) # 1st hidden layer
    n_hidden2 = int(h2_nodes) # 2nd hidden layer
    n_hidden3 = int(h3_nodes) # 3rd hidden layer
    n_output = 10   # output layer (0-9 digits)

    alpha = 1e-4
    n_iterations = 1000
    batch_size = 128
    dropout = 0.5

    X = tf.placeholder("float", [None, n_input])
    Y = tf.placeholder("float", [None, n_output])
    keep_prob = tf.placeholder(tf.float32)

    synapses = {
        'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
        'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
        'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
        'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
    }

    biases = {
        'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
        'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
        'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
        'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
    }

    # step 1 - guess stage - dot product of features for every layer
    layer_1 = tf.add(tf.matmul(X, synapses['w1']), biases['b1'])
    layer_2 = tf.add(tf.matmul(layer_1, synapses['w2']), biases['b2'])
    layer_3 = tf.add(tf.matmul(layer_2, synapses['w3']), biases['b3'])
    layer_drop = tf.nn.dropout(layer_3, keep_prob)
    output_layer = tf.matmul(layer_3, synapses['out']) + biases['out']

    # step 2 - guess stage - activation function of dot products
    # a standard tf gradient descent math function - calculates all derivatives using sigmoids
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # step 3 - training stage - cost function
    correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # tensorflow runtime
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # step 4 - weight adjustment - based on cross-entropy (gradient descent)
    # train on mini batches
    for i in range(n_iterations):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

        # loss and accuracy (per minibatch)
        if i%100==0:
            minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
            print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

            test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})


    test_accuracy = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob:1.0})
    print("\nAccuracy on test set:", test_accuracy)


    # Changing the 'Image.open( #filename# )' to any image in the project folder will change the image being classified
    # Note that the image must be 28x28 pixels : a good tool to make such digits is at https://www.pixilart.com/draw
    # Remember to make the background of your image white (not transparent) and the writing itself in greyscale colours
    img = np.invert(Image.open("test_img4.png").convert('L')).ravel()

    prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [img]})
    print ("Prediction for test image:", np.squeeze(prediction))


    correct_prediction = "3"

    if str(np.squeeze(prediction)) == str(correct_prediction):
        # when correct prediction
        m = "o"
    else:
        # when incorrect prediction
        m = "X"

    accuracyDict[str(n_hidden1)] = n_hidden1, test_accuracy, m

# even steps means next layers will always have whole nodes as nodes halve every layer
for i in range(10,1000,20):
    hiddenlayer1set = i
    hiddenlayer2set = round(i/2)
    hiddenlayer3set = round(hiddenlayer2set / 2)
    print(" ")
    print(" ")
    print(" NEW SETTINGS ---------------------------------------------------------------------------------------")
    print("Current settings: " + str(hiddenlayer1set) + " " + str(hiddenlayer2set) + " " + str(hiddenlayer3set))
    print(" ")
    runVarNodes(hiddenlayer1set,hiddenlayer2set,hiddenlayer3set)
    print(accuracyDict)

print(accuracyDict)


# need to get matplotlib library
import matplotlib.pyplot as plt

# repackage data into array-like for matplotlib
data = {"x":[], "y":[], "label":[], "m":[]}
for label, coord in accuracyDict.items():
    data["x"].append(coord[0])
    data["y"].append(coord[1])
    data["label"].append(label)
    data["m"].append(coord[2])


# display scatter plot data
plt.figure(figsize=(10,8))
plt.title('Accuracy of Classifier - Nodes in First Hidden Layer', fontsize=20)
plt.xlabel('Nodes in First Hidden Layer', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.scatter(data["x"], data["y"], marker = str(data["m"][0]))

# add labels
for label, x, y in zip(data["label"], data["x"], data["y"]):
    plt.annotate(label, xy = (x, y))

plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

# launches in python shell
plt.show()

