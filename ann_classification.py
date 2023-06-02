# Artificial Neural Network implementation using Tensorflow,Keras - Classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.layers import Dense
from keras.layers import InputLayer
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data,columns=cancer.feature_names)

X = df
y = pd.Series(cancer.target)
# Make train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle=True, random_state=2)

ANN_model = Sequential()

ANN_model.add(InputLayer(input_shape=(30, )))
# No hidden layers
ANN_model.add(Dense(1, activation='sigmoid'))

ANN_model.compile(optimizer=keras.optimizers.Adam,loss='binary_crossentropy',metrics=['accuracy'])

history = ANN_model.fit(X_train, y_train,epochs=10, batch_size=32,validation_split=0.2,shuffle=False)
test_loss, test_acc = ANN_model.evaluate(X_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
