import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Conv1D
from keras.layers import concatenate
from keras.models import Model

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from numpy import asarray

if __name__ == '__main__':
    # df = pd.read_csv('final.csv')
    # df = pd.read_csv('pex.csv')
    df = pd.read_csv('res2.csv')
    # get post_content
    max_size = 8000
    # max_size = 700
    df = df[pd.notnull(df['post_content'])]
    docs = df['post_content'].tolist()
    print("Docs: ", len(docs))

    gaps = np.array(df['post_gap'])
    train_gaps = gaps[:max_size]
    test_gaps = gaps[max_size:]

    # convert dialogue_act to binary array
    dialogue_acts = df['dialogue_acts'].tolist()
    labels = [acts.split(', ') for acts in dialogue_acts]

    train_docs = docs[:max_size]
    train_labels = labels[:max_size]

    test_docs = docs[max_size:]
    test_labels = labels[max_size:]

    print(test_docs[0:5])

    conversation_end = df['conversation_end']
    encoder = LabelEncoder()
    encoder.fit(conversation_end)
    encoded_Y = encoder.transform(conversation_end)
    # print(encoded_Y, len(encoded_Y))
    train_Y = encoded_Y[:max_size]
    test_Y = encoded_Y[max_size:]

    # prepare Tokenizer
    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    print("vocab_size: ", vocab_size)

    train_encoded_docs = t.texts_to_sequences(train_docs)
    test_encoded_docs = t.texts_to_sequences(test_docs)

    # pad documents to a max length of longest word
    MAX_LENGTH = max(len(d.split()) for d in docs)
    print("max_length: ", MAX_LENGTH)

    train_padded_docs = pad_sequences(train_encoded_docs, maxlen=MAX_LENGTH, padding='post')
    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=MAX_LENGTH, padding='post')

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

    # Create model
    # model = Sequential()
    # model.add(Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False))
    # model.add(Flatten())
    # model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #
    # print(model.summary())
    #
    # model.fit(train_padded_docs, train_Y, epochs=10, verbose=2, batch_size=64)
    # loss, accuracy = model.evaluate(test_padded_docs, test_Y, verbose=0)
    # print('Accuracy: %f' % (accuracy * 100), "Loss: ", loss)
    #
    # predictions = model.predict(test_padded_docs)
    #
    # test = [1 if x > 0.3 else 0 for i in predictions for x in i]
    # print(test)
    #
    # for idx, i in enumerate(predictions):
    #     # print(idx, i)
    #     for x in i:
    #         if x > 0.2:
    #             print(idx+8030, test_docs[idx], " x: ", x)
    #
    # print("Confusion matrix: \n", confusion_matrix(test_Y, test))

    ###########################################################################################

    # with post gap
    sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=MAX_LENGTH,
                                trainable=False)
    embedding_sequences = embedding_layer(sequence_input)
    x = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding_sequences)
    # x = Dropout(0.5)(x)
    # x = Flatten()(embedding_sequences)
    x = Flatten()(x)

    # gaps = gaps.reshape(gaps.shape[0], 1, 1)

    # Second input (post_gap)
    visible2 = Input(shape=(1,), dtype='float32')
    # conv21 = Conv1D(filters=100, kernel_size=4, activation='relu', input_shape=(1000, ))(visible2)
    # flat2 = Flatten()(visible2)

    merge = concatenate([x, visible2])

    # outputs = Dense(units=5, activation='sigmoid')(x)
    # model = Model(sequence_input, outputs)

    output = Dense(units=1, activation='sigmoid')(merge)
    model = Model(inputs=[sequence_input, visible2], outputs=output)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    print(model.summary())

    model.fit([train_padded_docs, train_gaps], train_Y, epochs=2, verbose=2, batch_size=32)
    loss, accuracy = model.evaluate([test_padded_docs, test_gaps], test_Y, verbose=0)
    print('Accuracy: %f' % (accuracy * 100), "Loss: ", loss)

    predictions = model.predict([test_padded_docs, test_gaps])
    # print(predictions)

    test = [1 if x > 0.011 else 0 for i in predictions for x in i]
    print(test)

    for idx, i in enumerate(predictions):
        # print(idx, i)
        for x in i:
            if x > 0.011:
                print(idx + 8030, test_docs[idx], " x: ", x)

    print("Confusion matrix: \n", confusion_matrix(test_Y, test))


    ###########################################################################################



