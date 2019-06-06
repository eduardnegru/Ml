import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from matplotlib import pyplot as plt
import nltk
import pickle
from nltk.tokenize import ToktokTokenizer
from nltk.corpus import inaugural, stopwords
from pathlib import Path
import argparse
import time
import re
import json
from nltk import ngrams
from tqdm import tqdm
from collections import Counter
from nltk import ngrams

CLI=argparse.ArgumentParser()
CLI.add_argument("--path", type=str, default="")
CLI.add_argument("--ngrams", type=int, default=0)
CLI.add_argument("--rows", type=int, default=None)
args, unknown = CLI.parse_known_args()

toxic = 0
non_toxic = 0
stopwords = stopwords.words("english")

def extract_hashtags(text):
	hashtags = re.findall(r"#(\w+)", text)

	if len(hashtags) == 0:
		return []

	return [[item] for item in hashtags]

full_text = " "

def parse_message(message, label):
	global toxic
	global non_toxic

	if label == 0:
		non_toxic += 1
	else:
		toxic += 1

	return message


dataset = pd.read_csv(args.path, encoding='latin-1', nrows=args.rows)

start = time.time()
result = [parse_message(text, label) for text, label in tqdm(zip(dataset["question_text"], dataset["target"]))]
end = time.time()

if args.ngrams != 0:
	full_text = full_text.join(result)
	ngram_counts = Counter(ngrams(full_text.split(), 3))
	most_common = ngram_counts.most_common(10)

print("Processing took " + str(end-start) + " seconds")

toxicity_count = {}
toxicity_count["toxic"] = toxic
toxicity_count["non_toxic"] = non_toxic

print("Toxic ", float(toxic * 100 / (toxic + non_toxic)))
print("Non-toxic ", float(non_toxic * 100 / (toxic + non_toxic)))