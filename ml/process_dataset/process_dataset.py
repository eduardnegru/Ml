import pandas as pd
import numpy as np
import argparse
import nltk
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

contractions = {
"ain't": "am not",
"aren't": "am not",
"can't": "cannot",
"can't've": "cannot have",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"isn't": "is not",
"mayn't": "may not",
"mightn't've": "might not have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"wasn't": "was not",
"weren't": "were not",
"won't": "will not",
"won't've": "will not have",
"wouldn't": "would not"
}

def process_words(text):
	words = nltk.word_tokenize(text)
	new_words = []
	for word in words:
		if word not in stopwords.words("english") and word.isalpha():
			new_words.append(word)

	return " ".join(new_words)

CLI=argparse.ArgumentParser()
CLI.add_argument("--path", type=str, default="")
CLI.add_argument("--output_file", type=str, default="output.csv")
args, unknown = CLI.parse_known_args()


dataset = pd.read_csv(args.path, encoding='latin-1')
print("Started")

results = []

for text, label in tqdm(zip(dataset["question_text"], dataset["target"])):

	if isinstance(text, str):
		words = text.split()
		# results.append([label, process_words(text)])
		if len(words) > 3:
			results.append([label, text])


df = pd.DataFrame(results, columns = ['target', 'question_text'])
df.to_csv("output", index=False)
