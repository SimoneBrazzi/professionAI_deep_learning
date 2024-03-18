import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import SimpleRNN

# activate only CPU
# tf.config.set_visible_devices([], 'GPU')

path = "/Users/simonebrazzi/datasets/IMDB Dataset.csv"
df = pd.read_csv(path)
df = df.sample(10000, random_state=1)

x = df.review.values
y = df.sentiment.values
le = LabelEncoder()
y = le.fit_transform(y)

xtrain, xtest, ytrain, ytest = train_test_split(
  x, 
  y,
  test_size=.2,
  random_state=1
)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(xtrain)

train_seq = tokenizer.texts_to_sequences(xtrain)
test_seq = tokenizer.texts_to_sequences(xtest)
vocabulary = len(tokenizer.word_index) + 1
maxlen = len(max(train_seq, key=len))
padded_train_sequences = pad_sequences(train_seq, maxlen=maxlen)
padded_test_sequences = pad_sequences(test_seq, maxlen=maxlen)

clear_session()
model_RNN = Sequential()
model_RNN.add(Embedding(
  input_dim=vocabulary,
  output_dim=128,
  input_shape=(maxlen,)
  ))
model_RNN.add(SimpleRNN(64))
model_RNN.add(Dense(1, activation="sigmoid"))

model_RNN.compile(
  optimizer="rmsprop",
  loss="binary_crossentropy",
  metrics=["accuracy"]
)

start = time.time()
model_RNN.fit(
  padded_train_sequences,
  ytrain,
  epochs=5,
  batch_size=256,
  validation_split=.2,
  verbose=1
)

end = time.time()

print(f"Training time: {end - start}")

model_RNN.evaluate(padded_test_sequences, ytest)
