'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

# Import MINST data
# import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import os
import numpy as np
import tensorflow as tf

class DataSet(object):
  def __init__(self, images, labels):
    self._num_examples = images.shape[0]
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


# Parameters
#learning_rate = 0.001
learning_rate = 0.005
training_iters = 300000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 25*25)
n_channels = 3
n_classes = 5 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units
filename = "5class.bin"
#filename = "5class.bin"

curr_dir = os.path.dirname(os.path.realpath(__file__))
data = np.fromfile(os.path.join(curr_dir, filename), dtype='float64')
n_row = int(data[0])
n_col= int(data[1])
data = np.delete(data, [0, 1])
data = data.reshape((n_row, n_col))
print("data.shape {}".format(data.shape))

feature = data[:,1:]
label = data[:,0]

#normalize pixels
feature = feature.astype(np.float32)
feature = np.multiply(feature, 1.0 / 255.0)
print(feature)

#change y from (n_row, 1) to (n_row, n_class)
tmp = np.zeros((n_row, n_classes))
for i in range(n_row):
    curr_class = label[i]
    tmp[i, curr_class] = 1

label = tmp
print(label)
print(label.shape)

train_split = int(n_row * 0.6)
test_split = int(n_row * 0.8)

x_train = feature[:train_split]
x_valid = feature[train_split:test_split]
x_test = feature[test_split:]
y_train = label[:train_split]
y_valid = label[train_split:test_split]
y_test = label[test_split:]

print("x_train.shape {}".format(x_train.shape))
print("x_valid.shape {}".format(x_valid.shape))
print("x_test.shape {}".format(x_test.shape))
print("y_train.shape {}".format(y_train.shape))
print("y_valid.shape {}".format(y_valid.shape))
print("y_test.shape {}".format(y_test.shape))

x_train = np.reshape(x_train, (x_train.shape[0], n_input, n_channels))
x_valid = np.reshape(x_valid, (x_valid.shape[0], n_input, n_channels))
x_test = np.reshape(x_test, (x_test.shape[0], n_input, n_channels))

print("x_train.shape {}".format(x_train.shape))
print("x_valid.shape {}".format(x_valid.shape))
print("x_test.shape {}".format(x_test.shape))
print("y_train.shape {}".format(y_train.shape))
print("y_valid.shape {}".format(y_valid.shape))
print("y_test.shape {}".format(y_test.shape))

class DataSets(object):
    pass
data_sets = DataSets()
data_sets.train = DataSet(x_train, y_train)
data_sets.validation = DataSet(x_valid, y_valid)
data_sets.test = DataSet(x_test, y_test)

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input, n_channels])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create model
def conv2d(img, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1, 1, 1, 1], padding='SAME'),b))

def max_pool(img, k):
    return tf.nn.max_pool(img, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

def conv_net(_X, _weights, _biases, _dropout):
    # Reshape input picture
    _X = tf.reshape(_X, shape=[-1, 28, 28, 3])

    # Convolution Layer
    conv1 = conv2d(_X, _weights['wc1'], _biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = max_pool(conv1, k=2)
    # Apply Dropout
    conv1 = tf.nn.dropout(conv1, _dropout)

    # Convolution Layer
    conv2 = conv2d(conv1, _weights['wc2'], _biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = max_pool(conv2, k=2)
    # Apply Dropout
    conv2 = tf.nn.dropout(conv2, _dropout)

    # Fully connected layer
    dense1 = tf.reshape(conv2, [-1, _weights['wd1'].get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1, _weights['wd1']), _biases['bd1'])) # Relu activation
    dense1 = tf.nn.dropout(dense1, _dropout) # Apply Dropout

    # Output, class prediction
    out = tf.add(tf.matmul(dense1, _weights['out']), _biases['out'])
    return out

# Store layers weight & bias
weights = {
    'wc1': tf.Variable(tf.random_normal([5, 5, 3, 32])), # 5x5 conv, 1 input, 32 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])), # 5x5 conv, 32 inputs, 64 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])), # fully connected, 7*7*64 inputs, 1024 outputs
    'out': tf.Variable(tf.random_normal([1024, n_classes])) # 1024 inputs, 10 outputs (class prediction)
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1

    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = data_sets.train.next_batch(batch_size)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            print("Valid Accuracy:", sess.run(accuracy, feed_dict={x: data_sets.validation.images, y: data_sets.validation.labels, keep_prob: 1.}))
            loss_val = sess.run(cost, feed_dict={x:data_sets.validation.images, y:data_sets.validation.labels, keep_prob: 1.})
            print("Valid Loss: {}".format(loss_val))
        step += 1
    print("Optimization Finished!")
    # Calculate accuracy for 256 data_sets test images
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: data_sets.test.images, y: data_sets.test.labels, keep_prob: 1.}))
