from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D

from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

import pandas as pd
import numpy as np
from numpy import asarray
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import hamming_loss

if __name__ == '__main__':
    seed = 7
    np.random.seed(seed)
    df = pd.read_csv('res2.csv')
    # compute_post_gap(df)

    # get post_content
    max_size = 700
    cdf = df
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
    train_y = np.array(Y)

    Y2 = mlb.fit_transform(test_labels)
    Y2 = Y2.tolist()
    test_y = np.array(Y2)

    # get quoted_posts
    cdf = cdf[pd.notnull(cdf['quoted_post'])]
    quoted_posts = cdf['quoted_post'].tolist()
    print("Quoted posts: ", len(quoted_posts))

    train_qp = quoted_posts[:max_size]
    test_qp = quoted_posts[max_size:]

    all_docs = quoted_posts + docs

    # prepare Tokenizer
    t = Tokenizer()
    # t.fit_on_texts(docs)
    t.fit_on_texts(all_docs)
    vocab_size = len(t.word_index) + 1
    print("vocab_size: ", vocab_size)

    train_encoded_docs = t.texts_to_sequences(train_docs)
    train_encoded_qp = t.texts_to_sequences(train_qp)

    test_encoded_docs = t.texts_to_sequences(test_docs)
    test_encoded_qp = t.texts_to_sequences(test_qp)

    # pad documents to a max length of longest word
    # MAX_LENGTH = max(len(d.split()) for d in docs)
    MAX_LENGTH = max(len(d.split()) for d in all_docs)
    # for d in docs:
    #     print(len(d.split()))
    print("max_length: ", MAX_LENGTH)
    # np.set_printoptions(threshold=np.inf)
    train_padded_docs = pad_sequences(train_encoded_docs, maxlen=MAX_LENGTH, padding='post')
    test_padded_docs = pad_sequences(test_encoded_docs, maxlen=MAX_LENGTH, padding='post')

    train_padded_qp = pad_sequences(train_encoded_qp, maxlen=MAX_LENGTH, padding='post')
    test_padded_qp = pad_sequences(test_encoded_qp, maxlen=MAX_LENGTH, padding='post')

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

    # Layer 1 - input layer using word embeddings
    sequence_input = Input(shape=(MAX_LENGTH,), dtype='int32')
    embedding_layer = Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=MAX_LENGTH, trainable=False)
    embedding_sequences = embedding_layer(sequence_input)
    x = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding_sequences)
    # print(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    outputs = Dense(units=5, activation='sigmoid')(x)
    model = Model(sequence_input, outputs)

    # Second input - quoted_posts
    sequence_input2 = Input(shape=(MAX_LENGTH, ), dtype='int32')
    embedding_layer2 = Embedding(vocab_size, dim_size, weights=[embedding_matrix], input_length=MAX_LENGTH,
                                trainable=False)
    embedding_sequences2 = embedding_layer2(sequence_input2)
    y = Conv1D(filters=100, kernel_size=4, activation='relu')(embedding_sequences2)
    y = Dropout(0.5)(y)
    y = Flatten()(y)

    merge = concatenate([x, y])

    outputs = Dense(units=5, activation='sigmoid')(merge)
    model = Model([sequence_input, sequence_input2], outputs)

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    # summarize the model
    print(model.summary())
    # fit the model
    # print(type(padded_docs), type(labels))
    print("Building model...")
    model.fit([train_padded_docs, train_padded_qp], train_y, epochs=5, verbose=2, batch_size=8)
    # model.fit(train_padded_docs, train_y, validation_data=[test_padded_docs, test_y], epochs=20, verbose=0, batch_size=64)
    loss, accuracy = model.evaluate([train_padded_docs, train_padded_qp], train_y, verbose=1)
    print('1: Accuracy: %f' % (accuracy * 100), "Loss: ", loss)
    loss, accuracy = model.evaluate([test_padded_docs, test_padded_qp], test_y, verbose=1)
    print('2: Accuracy: %f' % (accuracy * 100), "Loss: ", loss)

    text = ['bakit ka ba malungkot?', 'hahaha', 'sana andito siya...', 'nasaan ka na ba?', 'masaya mabuhay!']
    wow = np.array(t.texts_to_sequences(text))
    print(wow.shape)
    wow2 = pad_sequences(wow, maxlen=MAX_LENGTH, padding='post')
    # predictions = model.predict(wow2)
    # predictions = model.predict(test_padded_docs)
    predictions = model.predicts([test_padded_docs, test_padded_qp])
    print("Pred: ", type(predictions), " test_y: ", type(test_y))
    pred_binary = (predictions > 0.5).astype(int)
    print("pred_binary: ", type(pred_binary), "\n", pred_binary)
    y_classes = predictions.argmax(axis=-1)
    print(list(mlb.classes_))
    print(predictions)
    # print("y_classes: ", y_classes)
    # print(df['dialogue_acts'].head(1))
    print("Hamming loss: ", hamming_loss(test_y, pred_binary))
    # print(train_y[0], test_y[0])