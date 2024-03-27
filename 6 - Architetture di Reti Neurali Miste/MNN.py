# Setup
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from textwrap import wrap
import pickle

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Embedding
from tensorflow.keras.applications import VGG16
from tensorflow.keras.backend import clear_session
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from nltk.translate.bleu_score import sentence_bleu

def readImage(path, img_size=224):
  img = load_img(path, color_mode="rgb", target_size=(img_size, img_size))
  img = img_to_array(img)
  img = img / 255.
  
  return img

def display_images(x):
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)

    for i, ax in enumerate(axes.flat):
        image = readImage(f"{path}Images/{x.image[i]}")  # Assuming readImage function is defined elsewhere
        ax.imshow(image)
        ax.set_title("\n".join(wrap(x.caption[i], 20)), loc="right")
        ax.axis("off")
    
# Dataset
path = "/Users/simonebrazzi/datasets/flickr8k/"
df = pd.read_csv(path + "captions.txt")
# df.head()
df = df.sample(1000, random_state=42)

display_images(df)
plt.show()

# Transger Learning
vgg16 = VGG16()
vgg16 = Model(
  inputs=vgg16.inputs,
  outputs=vgg16.layers[-2].output
  )
vgg16.summary()


"""
image_features = {}
for name in os.listdir(f"{path}Images/"):
  filename = f"{path}Images/{name}"
  image = load_img(filename, target_size=(224, 224))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = preprocess_input(image)
  feature = vgg16.predict(image, verbose=0)
  image_features[name] = feature.flatten()
"""
with open("/Users/simonebrazzi/model/vgg16_MRNN_image_features.pkl", "rb") as f:
  image_features = pickle.load(f)
image_features["1000268201_693b08cb0e.jpg"].shape

# Captions
def text_preprocessing(data):
  
  data["caption"] = data.caption.apply(lambda x: x.lower())
  data["caption"] = data.caption.apply(lambda x: " ".join([word for word in x.split() if len(word)>1]))
  data["caption"] = "startseq " + data.caption + " endseq"
  return data

df = text_preprocessing(df)
captions = df.caption.to_list()

images, filename_index = [], []
for i, filename in enumerate(df.image):
  if filename in image_features.keys():
    images.append(image_features[filename])
    filename_index.append(i)

filenames = df.image.iloc[filename_index].values
captions = df.caption.iloc[filename_index].values
images = np.array(images)

#filenames[0], captions[0], images[0].shape

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(captions)
sequences = tokenizer.texts_to_sequences(captions)
vocab_size = len(tokenizer.word_index) + 1
maxlen = len(max(sequences, key=len))
#maxlen, vocab_size

val_size = int(len(sequences) * 0.2)
test_size = int(len(sequences) * 0.2)

def split_test_val_train(sequences, val_size, test_size):
  return (sequences[:test_size], sequences[test_size:test_size+val_size], sequences[test_size+val_size:])

txt_test, txt_val, txt_train = split_test_val_train(sequences, val_size, test_size)
img_test, img_val, img_train = split_test_val_train(images, val_size, test_size)
filename_test, filename_val, filename_train = split_test_val_train(filenames, val_size, test_size)

def prepare_data(sequences, images):
  
  xseq, ximg, yseq = [], [], []
  for seq, img in zip(sequences, images):
    for i in range(1, len(seq)):
      seq_in, seq_out = seq[:i], seq[i]
      seq_in = pad_sequences([seq_in], maxlen=maxlen).flatten()
      seq_out = to_categorical(seq_out, num_classes = vocab_size)
      xseq.append(seq_in)
      ximg.append(img)
      yseq.append(seq_out)
  
  xseq = np.array(xseq)
  ximg = np.array(ximg)
  yseq = np.array(yseq)
  print(f"\n# captions: {len(xseq)}")
  print(f"# images: {len(ximg)}")
  print(f"\nShapes: \n{xseq.shape} {ximg.shape} {yseq.shape}")
  return (xseq, ximg, yseq)

xseq_train, ximg_train, yseq_train = prepare_data(txt_train, img_train)
xseq_val, ximg_val, yseq_val = prepare_data(txt_val, img_val)
xseq_test, ximg_test, yseq_test = prepare_data(txt_test, img_test)

# Model
clear_session()
# image
input_image = Input(shape=(ximg_train.shape[1],), name="input_image")
cnn = Dense(256, activation="relu", name="cnn")(input_image)
#[i for i in dir(cnn) if not i.startswith("_")]
# text
input_txt = Input(shape=(maxlen,), name="input_txt")
embedding = Embedding(vocab_size, 64)(input_txt)
lstm = LSTM(256, name="lstm")(embedding)

# combined
cnn_lstm = layers.add([cnn, lstm])
cnn_lstm = Dense(256, activation="relu")(cnn_lstm)
output = Dense(vocab_size, activation="softmax")(cnn_lstm)

model = Model(
  inputs=[input_image, input_txt],
  outputs=output
)
model.summary()

# Compile
model.compile(
  optimizer="adam",
  loss="categorical_crossentropy",
)

# Fit
model.fit(
  [ximg_train, xseq_train],
  yseq_train,
  epochs=5,
  batch_size=64,
  validation_data=([ximg_val, xseq_val], yseq_val),
  verbose=1
  )

model.evaluate(
  [ximg_test, xseq_test],
  yseq_test
)

# Valutazione con algortmo BLEU



