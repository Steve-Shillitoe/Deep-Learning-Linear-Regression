
import matplotlib.pyplot as plt
import math
import random
import tensorflow as tf
import numpy as np

# We're trying to map our model to the equation y = mx + c + random noise
num_steps = 30
x_set = np.arange(num_steps, dtype="float")
m = random.uniform(1,5)
c = random.uniform(1,5)
noise = (np.random.rand(num_steps)-0.5)*20
y_set = np.array([ m*x + c for x in x_set], dtype="float") + noise
# Let's plot our dataset
plt.scatter(x_set, y_set)
plt.show()

# Create a single layer with 1 input and 1 output
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([layer_0])

# Compile the model with mean_sqared_error and Adam optimizer
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# Fit the model to our dataset x_set and y_set
model.fit(x_set, y_set, epochs=30)

# Convert the prediction matrix into a 1D array
prediction = model.predict(x_set).flatten()
plt.xlabel("x")
plt.ylabel("y")
 
# Our initial dataset
plt.scatter(x_set, y_set)
# Model prediction
ax = plt.plot(prediction, 'r')
plt.show()