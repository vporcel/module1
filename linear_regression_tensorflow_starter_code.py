import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the train dataset
train_dataset = {
    "inputs":np.array(
        [0, 2, 1, -2, -1]
    ),
    "labels":np.array(
        [4,3,4,8,6]
    )
}

# Define the val dataset
val_dataset = {
    "inputs":np.array(
        [0.5, 2, 0, -1.5, -1.2]
    ),
    "labels":np.array(
        [3.5,3.3,4,8.5,6]
    )
}

# Tool function to plot the dataset
def plot_dataset(train_dataset, val_dataset):
    plt.scatter(train_dataset["inputs"], train_dataset["labels"], label="train")
    plt.scatter(val_dataset["inputs"], val_dataset["labels"], label="val")
    plt.legend()
    plt.show()

# Training parameters
epsilon = 0.01
epochs = 50

###### MODEL DEFINITION #####
#TODO :  Create placeholders for data feeding

#TODO :  Create model variables theta0 and theta1

#TODO :  Create prediction node (i.e theta0*x+theta1)

#TODO :  Create loss node

#TODO :  Define optimizer ops

###### /MODEL DEFINITION #####


with tf.Session() as sess:
    # Run the variable initializer
    sess.run(tf.global_variables_initializer())

    # Loop through epochs number
    for e in range(epochs):

        ###### TRAINING LOOP #####
        # TODO: Compute accuracy and loss for train dataset after all the optimisation execution are done
        ###### /TRAINING LOOP #####


        ###### EVALUATION LOOP #####
        # TODO: Compute accuracy and loss for val dataset after all the optimisation execution are done
        ###### /EVALUATION LOOP #####

        pass