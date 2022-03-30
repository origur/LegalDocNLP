from pywebio.input import textarea
from pywebio.output import put_text
import pywebio
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
import re

COEF = 1
MIN_FREQ = 5000
FRACTION_OF_SENTENCES = 0.1
CONSTANT_SENTANCES = 0
# number of sentences you get = FRACTION * (Total # of sentences) + CONSTANT

stop_words = stopwords.words('english')
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

def remove_stopwords(sen):
    sen_new = " ".join([i for i in sen if i not in stop_words])
    return sen_new

def clean_text(text):
    new_text = re.sub('(?<![\.\!\?\n])\n', '.\n', text)
    new_text = re.sub('(?<=[A-Za-z])\-\s(?=[a-z])', '', new_text)
    new_text = re.sub('(?<=([Nn][Oo]))\.\s(?=[0-9])', ' ', new_text)
    new_text = re.sub('\([A-Za-z]\)\s', '', new_text)
    return new_text

def weighted_sum(i, coed, max_words):
    global word_embeddings, word_freq
    div = 0
    s = []
    for w in i.split():
        if word_freq.get(w, [0, 1000000, 1000000])[1] < max_words:
            for i in range(coed-1):
                div += 1
                s.append(word_embeddings.get(w, np.zeros((100,))))
        div += 1
        s.append(word_embeddings.get(w, np.zeros((100,))))
    return sum(s)/div+0.001


def summ(text):
    
    global stop_words, word_embeddings, word_freq
    

    

    sentences = sent_tokenize(text)

    

    clean_sentences = [remove_stopwords(r.split()) for r in [s.lower() for s in pd.Series(sentences).str.replace("[^a-zA-Z]", " ")]]

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = weighted_sum(i, COEF, MIN_FREQ) # change this to get weighted words
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
    n = int(len(sentences)*FRACTION_OF_SENTENCES) + CONSTANT_SENTANCES
    lower_score = ranked_sentences[n][0]

    output = []
    for i, s in enumerate(sentences):
        if scores[i] >= lower_score:
            output.append(s)
    return "\n".join(output)


def summerizer():
    text = textarea("Input legal text here: ")

    new_text = summ(clean_text(text))

    put_text(new_text)


if __name__ == '__main__':
    pywebio.start_server(summerizer, port=80)