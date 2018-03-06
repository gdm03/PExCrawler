from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D

from keras.layers import Reshape
from sklearn.metrics import hamming_loss

from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

import pandas as pd
import numpy as np
from numpy import asarray
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    # df = pd.read_csv('pex2.csv')
    seed = 7
    np.random.seed(seed)
    df = pd.read_csv('res2.csv')

    # get post_content
    max_size = 8000
    df = df[pd.notnull(df['post_content'])]
    docs = df['post_content'].tolist()
    print("Docs: ", len(docs))

    # gaps = np.array(df['post_gap'].tolist())
    gaps = np.array(df['post_gap'])
    print(gaps, gaps.shape)

    train_gaps = gaps[:max_size]
    test_gaps = gaps[max_size:]

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
    MAX_LENGTH = max(len(d.split()) for d in docs)
    # for d in docs:
    #     print(len(d.split()))
    print("max_length: ", MAX_LENGTH)
    # np.set_printoptions(threshold=np.inf)
    train_padded_docs = pad_sequences(train_encoded_docs, maxlen=MAX_LENGTH, padding='post')
    # print(padded_docs)

    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=MAX_LENGTH, padding='post')

    print(train_padded_docs.shape, gaps.shape)
    # load the whole embedding into memory
    embeddings_index = dict()

    f = open('wiki.tl.vec', encoding='utf-8')
    # f = open('glove.twitter.27B.200d.txt', encoding='utf-8')
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
    dim_size = 300
    # dim_size = 200
    embedding_matrix = np.zeros((vocab_size, dim_size))

    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # First input
    sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False)
    embedding_sequences = embedding_layer(sequence_input)
    # print(embedding_sequences, embedding_sequences.shape)
    x = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding_sequences)
    # x = Dropout(0.5)(x)
    # x = Flatten()(embedding_sequences)
    x = Flatten()(x)

    # gaps = gaps.reshape(gaps.shape[0], 1, 1)

    # Second input (post_gap)
    visible2 = Input(shape=(1, ), dtype='float32')
    # conv21 = Conv1D(filters=100, kernel_size=4, activation='relu', input_shape=(1000, ))(visible2)
    # flat2 = Flatten()(visible2)

    merge = concatenate([x, visible2])

    # outputs = Dense(units=5, activation='sigmoid')(x)
    # model = Model(sequence_input, outputs)

    output = Dense(units=5, activation='sigmoid')(merge)
    model = Model(inputs=[sequence_input, visible2], outputs=output)
    # Second input

    # compile the model
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # summarize the model
    print(model.summary())
    # fit the model
    # print(type(padded_docs), type(labels))
    print("Building model...")
    # model.fit(train_padded_docs, train_y, epochs=30, verbose=2, batch_size=64)
    model.fit([train_padded_docs, train_gaps], train_y, epochs=20, verbose=2, batch_size=32)

    # model.fit(train_padded_docs, train_y, validation_data=[test_padded_docs, test_y], epochs=20, verbose=0, batch_size=64)
    loss, accuracy = model.evaluate([train_padded_docs, train_gaps], train_y, verbose=1)
    print('1: Accuracy: %f' % (accuracy * 100), "Loss: ", loss)
    loss, accuracy = model.evaluate([test_padded_docs, test_gaps], test_y, verbose=1)
    print('2: Accuracy: %f' % (accuracy * 100))

    text = ['bakit ka ba malungkot?', 'hahaha', 'sana andito siya...', 'nasaan ka na ba?', 'masaya mabuhay!']
    wow3 = np.array([8, 3, 5, 7, 2])
    wow = np.array(t.texts_to_sequences(text))
    print(wow.shape)
    wow2 = pad_sequences(wow, maxlen=MAX_LENGTH, padding='post')
    # predictions = model.predict([wow2, wow3])
    predictions = model.predict([test_padded_docs, test_gaps])
    y_classes = predictions.argmax(axis=-1)
    print(list(mlb.classes_))
    print(predictions)
    # print("y_classes (highest): ", y_classes)
    # print(df['dialogue_acts'].head(1))
    pred_binary = (predictions > 0.5).astype(int)
    print("HL: %f" % (hamming_loss(test_y, pred_binary) * 100))
    # print(train_y[0], test_y[0])