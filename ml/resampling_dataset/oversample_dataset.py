import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import random
import nltk
import argparse

CLI=argparse.ArgumentParser()
CLI.add_argument("--path", type=str, default="")
CLI.add_argument("--toxic_percent", type=float, default=0.4)
CLI.add_argument("--output_file", type=str, default="oversampled.csv")
args, unknown = CLI.parse_known_args()


embeddings_dict = {}
index_to_word = {}
word_to_index = {}

def load_word_embeddings(file):

    f = open(file)
    i = 0
    for line in tqdm(f):
        values = line.split(" ")
        word = values[0]
        coefs = np.asarray(values[1:], dtype='str')
        embeddings_dict[word] = coefs
        index_to_word[i] = word
        word_to_index[word] = i
        if i > 300000:
            break
        i+=1
    f.close()

def make_tokenizer(texts, len_voc):
    from keras.preprocessing.text import Tokenizer
    t = Tokenizer(num_words=len_voc)
    t.fit_on_texts(texts)
    return t

def change_sentence(neighbours_mat, sentence):
    words = nltk.word_tokenize(sentence)
    words=[word.lower() for word in words if word.isalpha()]
    new_sentence = []
    for word in words:
        try:
            new_sentence.append(index_to_word[random.choice(neighbours_mat[word_to_index[word]])])
        except:
            new_sentence.append(word)
    return " ".join(new_sentence)

dataset = pd.read_csv("../../train.csv")
dataset = dataset.iloc[['target', 'question_text']]
load_word_embeddings('/home/adrian/Desktop/quora/glove/glove.txt')

tokenizer = make_tokenizer(dataset['question_text'], 100000)
X = tokenizer.texts_to_sequences(dataset['question_text'])


synonyms_number = 5
word_number = 40000

embeddings_matrix = np.array([embeddings_dict[i] for i in list(embeddings_dict.keys())])
nn = NearestNeighbors(n_neighbors=synonyms_number).fit(embeddings_matrix)
neighbours_mat = nn.kneighbors(embeddings_matrix[0:word_number])[1]
synonyms = {x[0]: x[1:] for x in neighbours_mat}

toxic = 0
non_toxic = 0

for label in dataset["target"]:
	if label == 0:
		non_toxic += 1
	else:
		toxic += 1

toxic_final = args.toxic_percent * non_toxic / (1 - args.toxic_percent)

toxic_to_generate = toxic_final - toxic
generated = 0

messages =[]
toxic_messages = [text for text, label in zip(dataset['question_text'], dataset['target']) if label == 1]

index = 0

while generated < toxic_to_generate:

    messages.append([1, change_sentence(synonyms ,toxic_messages[index])])

    if index == len(toxic_messages) - 1:
        index = 0
    else:
        index += 1

    generated += 1

df = pd.DataFrame(messages, columns =['target', 'question_text'])

finaldf = pd.concat([dataset, df])

finaldf.to_csv(args.output_file, index=False)