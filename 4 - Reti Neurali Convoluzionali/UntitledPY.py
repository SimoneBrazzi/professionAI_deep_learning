import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(
  Embedding(
    input_dim=vocabulary_size,
    output_dim=2,
    input_length=max_length
  )
)
