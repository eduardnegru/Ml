import os
import numpy as np
import pandas as pd
import math
import pickle
import json
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import CuDNNLSTM, Dense, Bidirectional, LSTM, Input, Flatten
from sklearn.externals import joblib
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import time


# Data providers
batch_size = 128
embeddings_index = {}

# # Convert text to embeddings
def text_to_array(text):
	# print(text.split()[:30])

	empyt_emb = np.zeros(300)
	text = text.split()[:30]
	embeds = [embeddings_index.get(word, empyt_emb) for word in text]
	embeds+= [empyt_emb] * (30 - len(embeds))
	return np.array(embeds)

def load_word_embeddings(file):
	f = open(file)
	i = 0
	for line in tqdm(f):
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
		if i == 10:
			break
		i = i + 1
	f.close()


def batch_gen(train_df):
	n_batches = int(math.ceil(len(train_df) / batch_size))
	while True:
		train_df = train_df.sample(frac=1.)  # Shuffle the data.
		for i in range(n_batches):
			texts = train_df.iloc[i*batch_size:(i+1)*batch_size, 0]
			text_arr = np.array([text_to_array(text) for text in texts])
			yield text_arr, np.array(train_df["target"][i*batch_size:(i+1)*batch_size])


def build_model():
	model = Sequential()
	model.add(Dense(1024, input_shape=(30, 300), activation="relu"))
	model.add(Dense(1024, input_shape=(30, 300), activation="relu"))
	model.add(Flatten())
	model.add(Dense(1, activation="sigmoid"))
	model.summary()
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def model_train_and_save(model, validation_x, validation_y):
	generator = batch_gen(train_df)
	model.fit_generator(generator, epochs=20,
						steps_per_epoch=1000,
						validation_data=(validation_x, validation_y),
						verbose=True)
	model.save("model_simple_nn")



train_df = pd.read_csv("../../train.csv")
train_df, validation_df = train_test_split(train_df, test_size=0.2, shuffle=True)


validation_x = np.array([text_to_array(text) for text in tqdm(validation_df["question_text"][:3000])])
validation_y = np.array(validation_df["target"][:3000])

load_word_embeddings('/home/adrian/Desktop/quora/wiki/wiki.vec')

#build the layers and compile
model = build_model()

#fit the data and save to disk
model_train_and_save(model, validation_x, validation_y)
