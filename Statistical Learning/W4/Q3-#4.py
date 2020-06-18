import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
import pandas as pd

mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
df = pd.read_csv('ptb.train', header=None)

hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128


x = tf.placeholder('float', [None, n_chunks,chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network(x):
    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),
             'biases':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(x, n_chunks, 0)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    prediction = recurrent_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(df.train.num_examples / batch_size)):
                epoch_x, epoch_y = df.train.next_batch(batch_size)
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',
              accuracy.eval({x: df.test.images.reshape((-1, n_chunks, chunk_size)), y: df.test.labels}))


train_neural_network(x)


'''
Epoch 0 completed out of 10 loss: 183.533833001
Epoch 1 completed out of 10 loss: 53.2128913924
Epoch 2 completed out of 10 loss: 36.641087316
Epoch 3 completed out of 10 loss: 28.2334972355
Epoch 4 completed out of 10 loss: 23.5787885857
Epoch 5 completed out of 10 loss: 20.3254865455
Epoch 6 completed out of 10 loss: 17.0910299073
Epoch 7 completed out of 10 loss: 15.3585778594
Epoch 8 completed out of 10 loss: 12.5780420878
Epoch 9 completed out of 10 loss: 12.060161829
Accuracy: 0.9827
'''
