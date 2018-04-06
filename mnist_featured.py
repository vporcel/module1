import tensorflow as tf
import numpy as np

# Define parameters
epochs = 500
batch_size = 500

# Read dataset
X = np.load("MNIST/10k_sample_featured/X_HMEAN.npy")
Y = np.load("MNIST/10k_sample_featured/Y_HMEAN.npy")

# Split into train and val datasets
X_TRAIN = X[0:8000]
Y_TRAIN = Y[0:8000]

X_VAL = X[8000:10000]
Y_VAL = Y[8000:10000]

# Define placholders
inputs = tf.placeholder(shape=(None, 28), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(None, ), name="labels", dtype=tf.int64)

# Define layer
W = tf.get_variable(name="weights", shape=(28, 10))
B = tf.get_variable(name="bias", shape=(10))

# define logits node
logits = tf.matmul(inputs, W) + B
predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

# define loss and optimizer
one_hot_labels = tf.one_hot(labels, 10)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=logits))
optimizer = tf.train.RMSPropOptimizer(0.001).minimize(loss)

# define accuracy
print(labels.shape, predictions.shape)
accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(X[0].shape)
    for e in range(epochs):
        for train_step in range(int(len(X_TRAIN) / batch_size)):
            _, lo, ac = sess.run([optimizer, loss, accuracy],
                                 feed_dict={inputs: X_TRAIN[train_step * batch_size:(train_step + 1) * batch_size],
                                            labels: Y_TRAIN[train_step * batch_size:(train_step + 1) * batch_size]})

        for val_step in range(int(len(X_VAL) / batch_size)):
            _, l, acc = sess.run([optimizer, loss, accuracy],
                                 feed_dict={inputs: X_VAL[val_step * batch_size:(val_step + 1) * batch_size],
                                            labels: Y_VAL[val_step * batch_size:(val_step + 1) * batch_size]})
