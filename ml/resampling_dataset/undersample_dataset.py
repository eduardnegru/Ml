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

'''
	The goal of this program is to undersample the majority class of a dataset up to a given %.
'''

CLI=argparse.ArgumentParser()
CLI.add_argument("--path", type=str, default="")
CLI.add_argument("--non_toxic_percent", type=float, default=0.6)
CLI.add_argument("--output_file", type=str, default="undersampled.csv")
args, unknown = CLI.parse_known_args()


if 1 < float(args.non_toxic_percent) or 0 > float(args.non_toxic_percent):
	raise Exception("Percentage must be in 0,1 range")


dataset = pd.read_csv(args.path, encoding='latin-1')
rows, cols = dataset.shape

toxic = 0
non_toxic = 0

for label in dataset["target"]:
	if label == 0:
		non_toxic += 1
	else:
		toxic += 1

non_toxic_to_keep = args.non_toxic_percent * toxic / (1 - args.non_toxic_percent)

indexes = []

for i, label in enumerate(dataset["target"]):

	if non_toxic_to_keep > 0 and label == 0:
		non_toxic_to_keep -= 1
		indexes.append(i)

	if label == 1:
		indexes.append(i)


new_dataset = dataset.iloc[indexes]


new_dataset.to_csv(args.output_file, index=False)