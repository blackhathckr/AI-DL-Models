# Self-Organizing Maps
import numpy as np
from som import SOM

# generate some random data with 36 features
data1 = np.random.normal(loc=-.25, scale=0.5, size=(500, 36))
data2 = np.random.normal(loc=.25, scale=0.5, size=(500, 36))
data = np.vstack((data1, data2))

som = SOM(10, 10)  # initialize a 10 by 10 SOM
som.fit(data, 10000, save_e=True, interval=100)  # fit the SOM for 10000 epochs, save the error every 100 steps
som.plot_error_history(filename='images/som_error.png')  # plot the training error history

targets = np.array(500 * [0] + 500 * [1])  # create some dummy target values

# now visualize the learned representation with the class labels
som.plot_point_map(data, targets, ['Class 0', 'Class 1'], filename='images/som.png')
som.plot_class_density(data, targets, t=0, name='Class 0', colormap='Greens', filename='images/class_0.png')
som.plot_distance_map(colormap='Blues', filename='images/distance_map.png')  # plot the distance map after training

# predicting the class of a new, unknown datapoint
datapoint = np.random.normal(loc=.25, scale=0.5, size=(1, 36))
print("Labels of neighboring datapoints: ", som.get_neighbors(datapoint, data, targets, d=0))

# transform data into the SOM space
newdata = np.random.normal(loc=.25, scale=0.5, size=(10, 36))
transformed = som.transform(newdata)
print("Old shape of the data:", newdata.shape)
print("New shape of the data:", transformed.shape)