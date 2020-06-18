import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.neural_network import MLPClassifier
from mlxtend.data import mnist_data
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

X, y = mnist_data()
print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('1st row', X[0])

clf1 = MLPClassifier(solver='adam', alpha=0,learning_rate='adaptive',max_iter=20,learning_rate_init=0.001,activation='relu',
                   hidden_layer_sizes=(100,), random_state=1)
clf11 = MLPClassifier(solver='adam', alpha=1e-3,learning_rate='adaptive',max_iter=20,learning_rate_init=0.001,activation='relu',
                      hidden_layer_sizes=(100,), random_state=1)

clf2 = MLPClassifier(solver='adam', alpha=0,learning_rate='adaptive',max_iter=20,learning_rate_init=0.001,activation='relu',
                    hidden_layer_sizes=(300,), random_state=1)
clf22 = MLPClassifier(solver='adam', alpha=1e-3,learning_rate='adaptive',max_iter=20,learning_rate_init=0.001,activation='relu',
                     hidden_layer_sizes=(300,), random_state=1)


X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
clf1.fit(X_train, y_train)
confidence1 = clf1.score(X_test, y_test)

clf11.fit(X_train, y_train)
confidence11 = clf11.score(X_test, y_test)

clf2.fit(X_train, y_train)
confidence2 = clf2.score(X_test, y_test)

clf22.fit(X_train, y_train)
confidence22 = clf22.score(X_test, y_test)


print('confidence : ', confidence1)
print('coef.shape : ', [coef.shape for coef in clf1.coefs_])
print('predict_proba : ',clf1.predict_proba(X_test))

############################################################################

# Utilizing TensorFlow

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#Built the model for our Artificial Neural Network and set up the computation graph with TensorFlow.
def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output

#Setting up the training process
def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x)

