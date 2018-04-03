import tensorflow as tf

input = tf.placeholder(shape=(), name="input", dtype=tf.float32)

theta0 = tf.placeholder(shape=(), name="theta0", dtype=tf.float32)

theta1 = tf.placeholder(shape=(), name="theta1", dtype=tf.float32)

mul = theta0 * input

prediction = mul + theta1

with tf.Session() as sess:

    result = sess.run(prediction, feed_dict={input:1, theta1:5, theta0:-1.2})
    
    print(result)