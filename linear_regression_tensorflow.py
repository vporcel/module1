import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the train dataset
train_dataset = {
    "inputs":np.array(
        [0, 2, 1, -2, -1, -1.5, -2.8, 3, 4.1]
    ),
    "labels":np.array(
        [4,3,4,8,6,7,9,0,0.2]
    )
}

# Define the val dataset
val_dataset = {
    "inputs":np.array(
        [0.5, 2, 0, -1.5, -1.2]
    ),
    "labels":np.array(
        [3.5,4.3,4,8.5,8]
    )
}

# Tool function to plot the dataset
def plot_dataset(train_dataset, val_dataset):
    plt.scatter(train_dataset["inputs"], train_dataset["labels"], label="train")
    plt.scatter(val_dataset["inputs"], val_dataset["labels"], label="val")
    plt.legend()
    plt.show()

# Plot the dataset
plot_dataset(train_dataset, val_dataset)

# Training parameters
epsilon = 0.01
epochs = 50

# Model definition
# Create placeholders for data feeding
inputs = tf.placeholder(shape=(), name="inputs", dtype=tf.float32)
labels = tf.placeholder(shape=(), name="labels", dtype=tf.float32)

# Create model parameters theta0 and theta1
theta0 = tf.get_variable(shape=(), name="theta0")
theta1 = tf.get_variable(shape=(), name="theta1")

# Compute prediction (i.e theta0*x+theta1)
mul = theta0 * inputs
prediction = mul + theta1

# Compute loss
loss = tf.square((prediction-labels))

# Define optimizer (here classic gradient descent)
optimizer = tf.train.GradientDescentOptimizer(epsilon).minimize(loss)

with tf.Session() as sess:
    # Run the variable initializer
    sess.run(tf.global_variables_initializer())

    # Create arrays for loss and accuracy curves plots
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []

    # Loop through epochs number
    for e in range(epochs):

        ###### TRAINING LOOP #####

        # Initialise our variable to compute loss and accuracy on this val epoch
        train_accuracy = 0
        train_loss = 0

        # Loop through training data : Learning phase
        for idx, data in enumerate(train_dataset["inputs"]):

            # Compute the optimizer, parameters and values we need
            _, theta_0, theta_1, train_loss_, train_pred, label_ = sess.run([optimizer, theta0, theta1, loss, prediction, labels], feed_dict={inputs:data, labels:train_dataset["labels"][idx]})

            # Compute the loss for train, add the current example to the total epoch train loss
            train_loss += train_loss_

            # Compute the accuracy for train, add the current example to the total epoch train accuracy
            if np.absolute((train_pred - label_)) < 1:
                train_accuracy += 1

        # Print the results
        print("-----------------------------------------------")
        print("Epoch %f : ( %f , %f)" % (e, theta_0, theta_1))
        print("Train accuracy = %d %%" % (train_accuracy * 100 / (idx + 1)))
        print("Train loss = %f " % (train_loss))

        # Store the train results
        train_accuracies.append((train_accuracy * 100 / (idx + 1)))
        train_losses.append(train_loss)
        ###### /TRAIN LOOP #####


        ###### EVALUATION LOOP #####
        # Initialise our variable to compute loss and accuracy on this val epoch
        val_accuracy = 0
        val_loss = 0

        # Loop through evaluation data : Testing phase
        for idx, data in enumerate(val_dataset["inputs"]):

            # We compute only what we need to get our accuracy and loss. Never run optimizer within your val loop
            pred, label, loss_ = sess.run([prediction, labels, loss], feed_dict={inputs:data, labels:train_dataset["labels"][idx]})

            # Compute the loss for val, add the current example to the total epoch train loss
            val_loss += loss_

            # Compute the accuracy for train, add the current example value to the total epoch train accuracy
            if np.absolute((pred-label)) < 1:
                val_accuracy += 1

        # Print the val results
        print("Val accuracy = %d %%" % (val_accuracy*100/(idx+1)))
        print("Val loss = %f " % (val_loss))

        # Store the val results
        val_accuracies.append((val_accuracy*100/(idx+1)))
        val_losses.append(val_loss)
        ###### /EVALUATION LOOP #####

    # Plot losses
    plt.plot(range(epochs), val_losses, label="val")
    plt.plot(range(epochs), train_losses, label="train")
    plt.legend()
    plt.show()

    # Plot accuracies
    plt.plot(range(epochs), val_accuracies, label="val")
    plt.plot(range(epochs), train_accuracies, label="train")
    plt.legend()
    plt.show()