import nltk
from nltk.stem.lancaster import lancasterStemmer

stemmer = lancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle","rb") as f:
        words, labels, train, output = pickle.load(f)

except:
    words = []
    labels = []
    docs_a = []
    docs_b = []

for intent in data["intents"]:
    for patern in intent["paterns"]:
# stemmer will take words and get only root word like whats will be what
        wrds = nltk.word_tokanize(patern)
        words.extend(wrds)
        docs.append(patern)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        words = [stemmer.stem(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(wrds)))
        labels = sorted(labels)

# creating bag of words for training model
train = []
output = []
empty_out=[0 for _ in range(len(labels))]

for a, doc in enumerate(docs_a):
    bag = []

    wrds = [stemmer.stem(w) for w in doc]

# now marking one or 0 into our bag of words by running loop on out words

for w in words:
    if w in wrds:
        bag.append(1)
    else:
        bag.append(0)
        
output_row = empty_out[:]
output_row[labels.index(docs_b[x])] = 1

train.append(bag)
output.append(output_row)

train = numpy.array(training)
output = numpy.array(output)

with open("data.pickle","wb") as f:
    pickle.dump((words, labels, train, output),f)



tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train[0])])

# hidden layer 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)

# output layer
net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")

except:        
    model.fit(train, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokanize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i]= 1
    return numpy.array(bag)

def chatting():
    print("start taking with the agent")
while True:
    inp = input("You:")
    if inp.lower() == "quit":
        break
    result = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            
    print(random.choice(responses))

chatting()
