import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    df = pd.read_csv('res2.csv')

    max_size = 8000
    df = df[pd.notnull(df['post_content'])]
    docs = df['post_content'].tolist()

    count_vect = CountVectorizer()
    train_docs = count_vect.fit_transform(docs[:max_size])

    tfidf_transformer = TfidfTransformer()
    train_docs_tf = tfidf_transformer.fit_transform(train_docs)
    print("train_docs_tf: ", train_docs_tf.shape)

    conversation_end = df['conversation_end']
    encoder = LabelEncoder()
    encoder.fit(conversation_end)
    encoded_Y = encoder.transform(conversation_end)
    # print(encoded_Y, len(encoded_Y))
    train_Y = encoded_Y[:max_size]
    test_Y = encoded_Y[max_size:]

    test_docs = count_vect.transform(docs[max_size:])
    test_docs_tf = tfidf_transformer.transform(test_docs)
    print("test_docs_tf: ", test_docs_tf.shape)
    # test_docs = new_docs_tf.reshape(-1, 1)
    # print(test_docs)

    # classifier = BinaryRelevance(MultinomialNB())
    # classifier = BinaryRelevance(OneVsRestClassifier(LinearSVC(random_state=0)))
    # classifier = BinaryRelevance(RandomForestClassifier(n_estimators=100))
    # print("Building model...")
    # classifier.fit(train_docs_tf, train_y)
    # print("Done fitting model. Predicting...")
    # predictions = classifier.predict(test_docs_tf)

    # predictions = OneVsRestClassifier(LinearSVC(random_state=0)).fit(train_docs_tf, train_y).predict(test_docs_tf)
    classifier = GaussianNB()
    # train_docs_np = train_docs_tf.toarray()
    train_docs_np = np.asarray(train_docs)
    classifier.fit(train_docs_tf.toarray(), train_Y)
    predictions = classifier.predict(test_docs_tf.toarray())
    print(predictions)
    print(confusion_matrix(test_Y, predictions))

    # testing = RandomForestClassifier(max_depth=2, random_state=0)
    # testing = RandomForestClassifier(n_estimators=100)
    # testing.fit(train_docs_tf, train_y)
    # predictions = testing.predict(test_docs_tf)
    # print("Feature importances: ", testing.feature_importances_)
    # print(testing)
    #
    # accuracy = accuracy_score(test_y, predictions)
    # print("Accuracy: ", accuracy)
    # print(mlb.classes_)

    accuracy = accuracy_score(test_Y, predictions)
    print("Accuracy: ", accuracy)

