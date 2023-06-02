# Deep Belief Networks - Regression
import numpy as np
np.random.seed(1337)  # for reproducibility
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from dbn.tensorflow import SupervisedDBNRegression

# Loading dataset
boston = load_boston()
X, Y = boston.data, boston.target

# Splitting data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1337)

# Data scaling
min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)

# Training
regressor = SupervisedDBNRegression(hidden_layers_structure=[100],learning_rate_rbm=0.01,learning_rate=0.01,n_epochs_rbm=20,n_iter_backprop=200,batch_size=16,activation_function='relu')
regressor.fit(X_train, Y_train)

# Test
X_test = min_max_scaler.transform(X_test)
Y_pred = regressor.predict(X_test)
print('Done.\nR-squared: %f\nMSE: %f' % (r2_score(Y_test, Y_pred), mean_squared_error(Y_test, Y_pred)))

#-----------------------------------------------------------------------------------------------------

# Deep Belief Networks - Classification
from sklearn.model_selection import train_test_split
from dbn.tensorflow import SupervisedDBNClassification
import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score

digits = pd.read_csv("train.csv")
from sklearn.preprocessing import standardscaler
X = np.array(digits.drop(["label"], axis=1))
Y = np.array(digits["label"])
ss=standardscaler()
X = ss.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
classifier = SupervisedDBNClassification(hidden_layers_structure =[256, 256], learning_rate_rbm=0.05, learning_rate=0.1, n_epochs_rbm=10, n_iter_backprop=100, batch_size=32, activation_function='relu', dropout_p=0.2)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
print('\n Accuracy of Prediction: %f' % accuracy_score(x_test, y_pred))