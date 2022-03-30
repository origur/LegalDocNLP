#!/bin/env python
import numpy as np
import pandas as pd
import nltk
#nltk.download('punkt') run this the first time
#nltk.download('stopwords') run this the first time
import re
from nltk.tokenize import sent_tokenize
import networkx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import networkx as nx

stop_words = stopwords.words('english')

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def weighted_sum(i):
    global word_embeddings, word_freq
    div = 0
    s = []
    for w in i.split():
        if word_freq.get(w, [1000, 100000, 100000])[1] < 5000:
            div += 2
            s.append(word_embeddings.get(w, np.zeros((100,))))
            s.append(word_embeddings.get(w, np.zeros((100,))))
        div += 1
        s.append(word_embeddings.get(w, np.zeros((100,))))
    return sum(s)/div+0.001

f = open("thing3.txt", "r", encoding='utf-8')
text = f.read()
f.close()

sentences = sent_tokenize(text)

word_embeddings = {}
f = open('glove.6B.100d.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_embeddings[word] = coefs
f.close()

word_freq = {}
f = open('en_words_1_1-64.txt', encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    word_freq[word] = coefs
f.close()

clean_sentences = [remove_stopwords(r.split()) for r in [s.lower() for s in pd.Series(sentences).str.replace("[^a-zA-Z]", " ")]]

sentence_vectors = []
for i in clean_sentences:
    if len(i) != 0:
        v = weighted_sum(i) # change this to get weighted words
    else:
        v = np.zeros((100,))
    sentence_vectors.append(v)


sim_mat = np.zeros([len(sentences), len(sentences)])

for i in range(len(sentences)):
    for j in range(len(sentences)):
        if i != j:
            sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]
            
nx_graph = nx.from_numpy_array(sim_mat)
scores = nx.pagerank(nx_graph)

ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
lower_score = ranked_sentences[len(sentences)//10+5][0]


for i, s in enumerate(sentences):
    if scores[i] >= lower_score:
        print(s)
