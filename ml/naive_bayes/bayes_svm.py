from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
from pathlib import Path
import nltk
import pickle
from nltk.tokenize import ToktokTokenizer
import argparse
import time
from tqdm import tqdm

'''
'''
def plot_confusion_matrix(y_true, y_pred, classes,
						  normalize=False,
						  title=None,
						  cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	"""
	if not title:
		if normalize:
			title = 'Normalized confusion matrix'
		else:
			title = 'Confusion matrix, without normalization'

	# Compute confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Only use the labels that appear in the data

	classes = unique_labels(y_true, y_pred)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)
	# We want to show all ticks...
	ax.set(xticks=np.arange(cm.shape[1]),
		   yticks=np.arange(cm.shape[0]),
		   # ... and label them with the respective list entries
		   xticklabels=classes, yticklabels=classes,
		   title=title,
		   ylabel='True label',
		   xlabel='Predicted label')

	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, format(cm[i, j], fmt),
					ha="center", va="center",
					color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	return ax

# defined command line options
# this also generates --help and error handling
CLI=argparse.ArgumentParser()

CLI.add_argument("--classifier",type=str, default="bayes")
CLI.add_argument("--kernel",type=str, default="linear")
CLI.add_argument("--C",type=float, default=1.0)
CLI.add_argument("--degree",type=int, default=3)
CLI.add_argument("--test_size",type=int, default=None)
CLI.add_argument("--train_size",type=int, default=None)
CLI.add_argument("--test_file",type=str, default="")
CLI.add_argument("--train_file",type=str, default="")
CLI.add_argument("--gamma",type=str, default="1.0")
CLI.add_argument("--confusion_matrix",type=str, default="confusion_matrix.png")
CLI.add_argument("--cache",type=str, default="quora_cache")

# parse the command line
args, unknown = CLI.parse_known_args()
# access CLI options


toktok = ToktokTokenizer()
np.random.seed(42)

#pos_tag returns the part of speech(pos) for a list of words
#but it gives too many details about the pos. Lemmatize pos
#argument wants to know only if the word is verb, adjective, noun etc.
def get_wordnet_pos(tag):

	if tag.startswith('J'):
		return wn.ADJ
	elif tag.startswith('V'):
		return wn.VERB
	elif tag.startswith('N'):
		return wn.NOUN
	elif tag.startswith('R'):
		return wn.ADV
	else:
		return wn.NOUN

def train_test_classifier(train_x_tfidf, test_x_tfidf, train_y):

	if args.classifier == "bayes":
		Naive = naive_bayes.MultinomialNB()
		start = time.time()
		Naive.fit(train_x_tfidf, train_y)
		end  = time.time()
		print("Model fit took " + str(end - start) + " seconds")
		start = time.time()
		predictions = Naive.predict(test_x_tfidf)
		end = time.time()
		print("Model prediction took " + str(end - start) + " seconds")
	else:
		SVM = svm.SVC(C=float(args.C), kernel=args.kernel, degree=int(args.degree), gamma=float(args.gamma))

		start  = time.time()
		SVM.fit(train_x_tfidf, train_y)
		end = time.time()
		print("Model fit took " + str(end - start) + " seconds")
		start = time.time()
		predictions = SVM.predict(test_x_tfidf)
		end = time.time()
		print("Model prediction took " + str(end - start) + " seconds")
	return predictions

'''
	Process the training dataset for nlp. The parameter is the
	object returned from read_csv("/path/to/training/dataset")

	@param {DataFrame}
	@returns {array}
'''
def process_training_dataset(train_dataset):

	file = Path(args.cache)
	if not file.is_file():

		train_dataset["question_text"] = train_dataset["question_text"].map(lambda text: toktok.tokenize(text.lower()))

		parsedDataset = []
		for index, tokenizedText in tqdm(enumerate(train_dataset["question_text"])):
			words = []
			wordLemmatizer = WordNetLemmatizer()
			for word, posTag in pos_tag(tokenizedText):
				if word not in stopwords.words("english") and word.isalpha():
					words.append(wordLemmatizer.lemmatize(word,get_wordnet_pos(posTag)))

			parsedDataset.append(str(words))

		print("Writing to file")
		with open(args.cache, 'wb') as fp:
			pickle.dump(parsedDataset, fp)
	else:
		print("Reading from cached file")
		f = open(args.cache, 'rb')
		parsedDataset = pickle.load(f)
		f.close()
		parsedDataset = parsedDataset[:args.train_size]
	return parsedDataset

train_dataset = pd.read_csv(args.train_file, encoding='latin-1', nrows=args.train_size)
test_dataset = pd.read_csv(args.test_file, encoding='latin-1', nrows=args.test_size)

processed_train_text = process_training_dataset(train_dataset)

print("Finished processing")

train_x = processed_train_text
train_y = train_dataset['target']

test_x = test_dataset['question_text']
test_y = test_dataset['target']


Encoder = LabelEncoder()
train_y = Encoder.fit_transform(train_y)

vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(processed_train_text)

train_x_tfidf = vectorizer.transform(train_x)
test_x_tfidf = vectorizer.transform(test_x)

predictions = train_test_classifier(train_x_tfidf, test_x_tfidf, train_y)

print("Accuracy Score -> ", accuracy_score(test_y, predictions) * 100)
print("F1 Score -> ", f1_score(test_y, predictions, average="macro"))
print("Precision Score -> ", precision_score(test_y, predictions, average="macro"))
print("Recall Score -> ", recall_score(test_y, predictions, average="macro"))


plot_confusion_matrix(test_y, predictions, classes=[str(i) for i in range(0, 2)], normalize=False, title=args.confusion_matrix)
plt.savefig("confusion_matrix" + args.classifier)
plt.close()