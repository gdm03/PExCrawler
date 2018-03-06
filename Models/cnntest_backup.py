from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

import pandas as pd
import numpy as np
from numpy import asarray
from sklearn.preprocessing import MultiLabelBinarizer

def convert_labels_to_int(labels):
    target_names = ['Comment', 'Question', 'Backchannel', 'Expression', 'Others']
    d = dict(zip(target_names, range(0, 5)))
    enc_labels = []

    for label in labels.split(', '):
        enc_labels.append(d[label])

    return enc_labels

def check_wrong_spelling(dialogue_acts):
    i = 0
    j = 0
    h = 0
    ctr = 0
    for acts in dialogue_acts:
        ctr = ctr + 1
        if acts == 'Yes':
            i = i + 1
        elif acts == 'No':
            j = j + 1
        else:
            print(ctr)
            h = h + 1
    print(i, j, h)

def compute_post_gap(df):
    post_dates = df['post_time']
    df['post_gap'] = pd.to_datetime(df['post_time']) - pd.to_datetime(df['post_time']).shift(-1)

    for index, row in df.iterrows():
        if row['post_counter'] == 1:
            current_time = pd.to_datetime(row['post_time'])
            post_gap_int = 0
            print(row['post_counter'], row['post_time'], post_gap_int)
        else:
            # post_gap_dt = pd.to_datetime(row['post_time']) - pd.to_datetime(current_time)
            post_gap_dt = pd.to_datetime(row['post_time']) - current_time
            # print(row['post_counter'], row['post_time'], gap)
            post_gap_int = (post_gap_dt / np.timedelta64(1, 'D')).astype(int)
            print(row['post_counter'], row['post_time'], post_gap_dt, post_gap_int, type(row['post_counter']))

if __name__ == '__main__':
    df = pd.read_csv('pex.csv')
    compute_post_gap(df.head(15))

    # get post_content
    # docs = df['post_content'].head(1463).tolist()
    # docs = df[pd.notnull(df['post_content'])]['post_content'].head(1463).tolist()
    max_size = 8000
    df = df[pd.notnull(df['post_content'])]
    docs = df['post_content'].tolist()
    print("Docs: ", len(docs))

    # convert dialogue_act to binary array
    dialogue_acts = df['dialogue_acts'].tolist()
    labels = [acts.split(', ') for acts in dialogue_acts]

    train_docs = docs[:max_size]
    train_labels = labels[:max_size]

    test_docs = docs[max_size:]
    test_labels = labels[max_size:]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(train_labels)
    Y = Y.tolist()
    # print(Y)
    train_y = np.array(Y)
    # print(train_y)
    # print(list(mlb.classes_))

    Y2 = mlb.fit_transform(test_labels)
    Y2 = Y2.tolist()
    test_y = np.array(Y2)

    # prepare Tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    print("vocab_size: ", vocab_size)

    train_encoded_docs = t.texts_to_sequences(train_docs)
    # print(t.word_index)
    # print(encoded_docs)
    # print(train_encoded_docs)

    test_encoded_docs = t.texts_to_sequences(test_docs)

    # pad documents to a max length of longest word
    max_length = max(len(d.split()) for d in docs)
    # for d in docs:
    #     print(len(d.split()))
    print("max_length: ", max_length)
    # np.set_printoptions(threshold=np.inf)
    train_padded_docs = pad_sequences(train_encoded_docs, maxlen=max_length, padding='post')
    # print(padded_docs)

    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=max_length, padding='post')

    # load the whole embedding into memory
    embeddings_index = dict()

    # f = open('wiki.tl.vec', encoding='utf-8')
    f = open('glove.twitter.27B.200d.txt', encoding='utf-8')
    i = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        # i = i+1
        # print(i, word)
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    # wiki.tl.vec = 300, glove = 200
    # dim_size = 300
    dim_size = 200
    embedding_matrix = np.zeros((vocab_size, dim_size))

    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    e = Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=max_length, trainable=False)
    model.add(e)
    model.add(Flatten())
    model.add(Dense(units=5, input_dim=max_length, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    # print(type(padded_docs), type(labels))
    print("Building model...")
    model.fit(train_padded_docs, train_y, epochs=50, verbose=0, batch_size=64)
    # evaluate the model
    # loss, accuracy = model.evaluate(train_padded_docs, train_y, verbose=0)
    loss, accuracy = model.evaluate(test_padded_docs, test_y, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    text = ['bakit ka ba malungkot?', 'hahaha', 'sana andito siya...', 'nasaan ka na ba?', 'masaya mabuhay!']
    wow = np.array(t.texts_to_sequences(text))
    print(wow.shape)
    wow2 = pad_sequences(wow, maxlen=max_length, padding='post')
    prediction = model.predict(wow2)
    y_classes = prediction.argmax(axis=-1)
    print(prediction)
    print("y_classes: ", y_classes)
    print(df['dialogue_acts'].head(1))
    print(train_y[0], test_y[0])
    print(list(mlb.classes_))


