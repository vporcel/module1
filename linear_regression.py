train_data = [
    [0,4], # [x0, y0]
    [2,3], # [x1, y1] ...
    [1,4],
    [-2,8],
    [-1,6],
]

# Model parameters
theta0 = 0.1
theta1 = -0.05

# Training parameters
epsilon = 0.01
epochs = 50

# Partial derivative of the loss function w.r.t theta 0
def dl_dtheta0(x, y, theta0, theta1):
    return(-2*x*(y-theta0*x-theta1))

# Partial derivative of the loss function w.r.t theta 1
def dl_dtheta1(x, y, theta0, theta1):
    return(-2*(y-theta0*x-theta1))

# Loop through epochs number
for e in range(epochs):
    # Loop through training data
    for data in train_data:
        # Update Theta0 parameter
        theta0 = theta0 - epsilon*dl_dtheta0(data[0], data[1], theta0, theta1)
        # Update Theta1 parameter
        theta1 = theta1 - epsilon*dl_dtheta1(data[0], data[1], theta0, theta1)
    print("Epoch "+str(e)+" : ("+str(theta0)+", "+str(theta1)+")")