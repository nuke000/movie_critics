from scipy import sparse as sp_sparse
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
import pandas as pd
import numpy as np
import re

def words_only(text):
    tag_regexp = re.compile("<[^>]*>")
    regex = re.compile("[A-Za-z-]+")
    text = re.sub(tag_regexp, '', text)
    text = re.sub('\s+', ' ',text)
    text = re.sub(r'\\','', text)
    text = text.lower().strip()
    try:
        return " ".join(regex.findall(text))
    except:
        return ""

def BoW(words, words_to_index, dict_size):
    """
        words: a list of words
        dict_size: size of the dictionary
        return a vector which is a bag-of-words representation of 'text'
    """
    result_vector = np.array([0 for i in range(dict_size)])
    wti = np.array(words_to_index)
    
    for word in words:
        ind = np.where(wti == word)
        if len (ind) == 1 :
            result_vector[ind[0]] +=1
    return result_vector

class SentimentClassifier:
    def __init__(self):
        self._data = []
        self._datapath = 'https://github.com/mbburova/MDS/raw/main/sentiment.csv'
        self._model = RandomForestClassifier(n_estimators = 300, random_state=5, max_depth = 5)
        self._X_train = []
        self._X_test = []
        self._y_train = []
        self._y_test = []
        self._DICT_SIZE = 500
        self._WORDS_TO_INDEX = []
        self._accuracy = 0
    
    def fit_model(self, datapath = 'https://github.com/mbburova/MDS/raw/main/sentiment.csv') :
        self._data = pd.read_csv(datapath, index_col=0)
        self._datapath = datapath
        tag_regexp = re.compile("<[^>]*>")
        regex = re.compile("[A-Za-z-]+")
        self._data['cleaned_review'] = self._data['review'].apply(words_only)
        self._data['tokenized'] = self._data['cleaned_review'].apply(lambda x: x.split())
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._data['tokenized'],self._data['sentiment'], test_size=0.2, random_state = 5)
        
        nltk.download('stopwords')
        STOPWORDS = set(stopwords.words('english'))
        
        counter = Counter(self._X_train.sum())
        for word in list(counter):
            if word in STOPWORDS:
                del counter[word]
        
        words_counts =  counter
        self._WORDS_TO_INDEX = [word[0] for word in counter.most_common(self._DICT_SIZE)]
        
        X_train_bow = sp_sparse.vstack([sp_sparse.csr_matrix(BoW(text, self._WORDS_TO_INDEX, self._DICT_SIZE)) for text in self._X_train])
        X_test_bow = sp_sparse.vstack([sp_sparse.csr_matrix(BoW(text, self._WORDS_TO_INDEX, self._DICT_SIZE)) for text in self._X_test])
        
        self._model = RandomForestClassifier(n_estimators = 300, random_state=5, max_depth = 5)
        self._model = self._model.fit(X_train_bow, self._y_train)
        pred = self._model.predict(X_test_bow)
        self._accuracy = accuracy_score(self._y_test, pred)
        
        return self._model
    
    def predict(self, sample):
        X_sample = [words_only(sample).split()]
        #X_sample = [['i','must','say','it','s', 'perfect'],['i','feel','horrible']]
        X_sample_bow = sp_sparse.vstack([sp_sparse.csr_matrix(BoW(text, self._WORDS_TO_INDEX, self._DICT_SIZE)) for text in X_sample])
        pred = self._model.predict(X_sample_bow)
        return pred[0]
