import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# fix random seed for reproducibility
np.random.seed(7)

if __name__ == '__main__':
    # load the dataset but only keep the top n words, zero the rest
    df = pd.read_csv('res.csv')
    top_words = 5000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)

    # truncate and pad input sequences
    max_review_length = 500
    X_train = pad_sequences(X_train, maxlen=max_review_length)
    X_test = pad_sequences(X_test, maxlen=max_review_length)
    # create the model
    embedding_vecor_length = 32
    model = Sequential()
    model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(X_train, y_train, nb_epoch=3, batch_size=64)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))