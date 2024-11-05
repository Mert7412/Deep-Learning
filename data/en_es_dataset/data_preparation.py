import numpy as np
import pandas as pd
import string

data = pd.read_csv(r"en_es_dataset\data.csv")

def tokenize_sentences(data,sentence_lenght = 16):
    sentences = []
    translator = str.maketrans("","",string.punctuation)
    for sentence in data:
        temp_sentence = []
        
        sentence =  str.lower(sentence.translate(translator))
        sentence = "<bos> " + sentence + " <eos>" 
        sentence = sentence.split(" ")
        len_sentence = len(sentence)
        for i in range(sentence_lenght):
            if i < len_sentence:
                temp_sentence.append(sentence[i])
            else:
                temp_sentence.append("<padding>")

        sentences.append(temp_sentence)
    return sentences
    
en_sentences = tokenize_sentences(data["english"])
es_sentences = tokenize_sentences(data["spanish"])

def creating_vobulary(sentences):
    vocabulary = {}
    reversed_vocabulary = {}
    k=0
    for sentence in sentences:
        for word in sentence:
            if word not in vocabulary:
                vocabulary[word] = k
                reversed_vocabulary[k] = word
                k+=1

    return vocabulary,reversed_vocabulary

en_vocab , reversed_en_vocab = creating_vobulary(en_sentences)
es_vocab , reversed_es_vocab = creating_vobulary(es_sentences)

def index_data(data,vocab):
    sentences = []
    for sentece in data:
        temp = []
        for word in sentece:
            temp.append(vocab[word])
        sentences.append(temp)
    return np.array(sentences)

en_data = index_data(en_sentences,en_vocab)
es_data = index_data(es_sentences,es_vocab)
np.save("en_sentences.npy",en_data)
np.save("es_sentences.npy",es_data)

np.save("es_vocab.npy",es_vocab)
np.save("en_vocab.npy",en_vocab)

