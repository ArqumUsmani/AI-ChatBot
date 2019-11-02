import nltk
from nltk.stem.lancaster import lancasterStemmer

stemmer = lancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    data = json.load(file)

words = []
labels = []
docs = []

for intent in data["intents"]:
    for patern in intent["paterns"]:
        # stemmer will take words and get only root word like whats will be what
        wrds = nltk.word_tokanize(patern)
        words.extend(wrds)
        docs.append(patern)

        if intent["tag"] not in labels:
            labels.append(intent["tag"])