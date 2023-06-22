import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import WordNetLemmatizer
import itertools
from wordcloud import WordCloud
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from keras.models import Sequential,Model
from keras.layers import Dense,LSTM,GRU, SpatialDropout1D, Embedding
from keras.utils.np_utils import to_categorical
from joblib import dump, load
from keras import layers, Input
from keras.layers import Bidirectional, GRU, RepeatVector, Dense, TimeDistributed
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector, GRU
from keras.optimizers import SGD, Adam, RMSprop, Adamax
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, precision_score

def start():
    df = pd.read_csv('labeled_data.csv')
    #resumpling
    #################################################
    df_00 = df[df['class']==0]
    df_11 = df[df['class']==1]
    df_22 = df[df['class']==2]
    df_00_resumple = resample(df_00,replace=True,n_samples=3676,random_state=123)
    df_22_resumple = resample(df_22, replace=True, n_samples=5576, random_state=123)
    df_11_resumple = resample(df_11, replace=False, n_samples=10190, random_state=123)
    df_resampled = pd.concat([df_00_resumple, df_22_resumple, df_11_resumple])
    df=df_resampled
    for j, (i, row) in enumerate(df.iterrows()):
        df.at[i,'Unnamed: 0'] =j
    df = df.reset_index()
    #################################################
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    df['tidy_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")
    df.tidy_tweet = df.tidy_tweet.str.replace("[^a-zA-Z#]", " ")
    df.head(10)

    # tokenizing
    tokenized_tweet = df.tidy_tweet.apply(lambda x: x.split())
    # stemming
    stemmer = PorterStemmer()
    tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])
    # detokenizing
    for i in range(len(tokenized_tweet)):
        tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    df['tidy_tweet'] = tokenized_tweet
    tokenized_tweet = df['tidy_tweet'].apply(lambda x: x.split())  # tokenizing
    def add_label(twt):
        output = []
        for i, s in zip(twt.index, twt):
            output.append(TaggedDocument(s, ["tweet_" + str(i)]))
        return output
    labeled_tweets = add_label(tokenized_tweet)
    labeled_tweets[:6]

    #DOC2VEC
    model_d2v = Doc2Vec(dm=1, dm_mean=1, vector_size=70, window=5, negative=7, min_count=5, workers=32, alpha=0.1, seed = 23)
    model_d2v.build_vocab([i for i in labeled_tweets])
    model_d2v.train(labeled_tweets, total_examples= len(df['tidy_tweet']), epochs=20)
    docvec_arrays = np.zeros((len(tokenized_tweet), 70))
    for i in range(len(df)):
        docvec_arrays[i,:] = model_d2v.docvecs[i].reshape((1,70))
    docvec_df = pd.DataFrame(docvec_arrays)
    docvec_df.shape


    #KNN
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    X = docvec_df
    y = df['class']
    accuracy_list = []
    for fold_id, (train_index, test_index) in enumerate(rskf.split(X,y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # svc = svm.SVC(kernel='linear', C=1, probability=True).fit(X_train, y_train)
        # prediction = svc.predict_proba(X_test)
        # prediction_int = prediction[:,1] >= 0.3
        # prediction_int = prediction_int.astype(np.int)
        # f1_score(y_test, prediction_int, average='micro')

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))

    print("Accuracy:", np.mean(accuracy_list))

    #SIEĆ NEURONOWA
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1234)
    X = docvec_df
    y = df['class']
    for fold_id, (train_index, test_index) in enumerate(rskf.split(X,y)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        y_train=to_categorical(y_train, num_classes = 3, dtype='float32')
        y_test=to_categorical(y_test, num_classes = 3, dtype='float32')

        model = Sequential()
        model.add(Embedding(24783, 100, input_length=X_train.shape[1]))
        model.add(SpatialDropout1D(0.3))
        model.add(GRU(80, dropout=0.6, recurrent_dropout=0.6))
        model.add(Dense(3, activation='ReLU'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        epochs = 25
        batch_size = 128
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=epochs, batch_size=batch_size)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

        break
if __name__ == '__main__':
    start()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
