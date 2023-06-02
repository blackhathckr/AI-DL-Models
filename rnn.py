# Recurrent Neural Network
import tensorflow as tf
from keras.preprocessing import text_dataset_from_directory
from keras.layers import Dense
from keras.models import Sequential
from keras import Input
from keras.layers.experimental.preprocessing import TextVectorization
from keras.layers import Embedding
from keras.layers import LSTM

train_data = text_dataset_from_directory("./train")
test_data = text_dataset_from_directory("./test")

from keras.preprocessing import text_dataset_from_directory
from tf.strings import regex_replace

def prepareData(dir):
   data = text_dataset_from_directory(dir)
   return data.map(lambda text, label: (regex_replace(text, '<br />', ' '), label),)

train_data = prepareData('./train')
test_data = prepareData('./test')

for text_batch, label_batch in train_data.take(1):
  print(text_batch.numpy()[0])
  print(label_batch.numpy()[0]) # 0 = negative, 1 = positive

model = Sequential()
model.add(Input(shape=(1,), dtype="string"))

max_tokens = 1000
max_len = 100
vectorize_layer = TextVectorization(max_tokens=max_tokens,output_mode="int",output_sequence_length=max_len)

train_texts = train_data.map(lambda text, label: text)
vectorize_layer.adapt(train_texts)

model.add(vectorize_layer)

# Previous layer: TextVectorization
max_tokens = 1000

model.add(vectorize_layer)

model.add(Embedding(max_tokens + 1, 128))
# 64 is the "units" parameter, which is the
# dimensionality of the output space.
model.add(LSTM(64))

model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'],)

model.fit(train_data, epochs=10)