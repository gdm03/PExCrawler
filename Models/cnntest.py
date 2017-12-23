import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import MultiLabelBinarizer

def convert_labels_to_int(labels):
    target_names = ['Comment', 'Question', 'Backchannel', 'Expression', 'Others']
    d = dict(zip(target_names, range(0, 5)))
    enc_labels = []

    for label in labels.split(', '):
        enc_labels.append(d[label])

    return enc_labels

if __name__ == '__main__':
    df = pd.read_csv('pex.csv')
    docs = df['post_content'].head()

    # y_train_text = df['dialogue_acts'].str.split(', ').dropna().tolist()
    # # print(y_train_text)
    # mlb = MultiLabelBinarizer()
    # Y = mlb.fit_transform(y_train_text)
    # print(Y)
    # print(list(mlb.classes_))

    # one_hot_encoded = tf.one_hot(indices=[[0, 1], [2, 3], [0, 4]], depth=5)

    labels = df['dialogue_acts'].head(1)
    #x = [convert_labels_to_int(label) for label in labels]

        #print(one_hot_encoded)

    # one_hot_encoded = tf.one_hot(indices=[])
    with tf.Session():
        for x in labels:
            one_hot_encoded = tf.one_hot(indices=convert_labels_to_int(x), depth=5)
            print(one_hot_encoded.eval())
            print("------")

    t = Tokenizer()
    t.fit_on_texts(docs)
    vocab_size = len(t.word_index) + 1
    print(vocab_size)

    # define documents
    docs2 = ['Well done!',
            'Good work',
            'Great effort',
            'nice work',
            'Excellent!',
            'Weak',
            'Poor effort!',
            'not good',
            'poor work',
            'Could have done better.']
    # define class labels
    labels2 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    # integer encode the documents
    vocab_size = 50
    encoded_docs = [one_hot(d, vocab_size) for d in docs2]
    print(encoded_docs)
    # for d in docs:
    #     print(one_hot(d, vocab_size))
    #     print("-----------------")

    # pad documents to a max length of 4 words
    max_length = 4
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
    print(padded_docs)

