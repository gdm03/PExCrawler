from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd
import graphviz

if __name__ == '__main__':
    df = pd.read_csv('res2.csv')

    max_size = 8000
    df = df[pd.notnull(df['post_content'])]
    docs = df['post_content'].tolist()

    # count_vect = CountVectorizer()
    # train_docs = count_vect.fit_transform(docs[:max_size])
    #
    # tfidf_transformer = TfidfTransformer()
    # train_docs_tf = tfidf_transformer.fit_transform(train_docs)
    # print("train_docs_tf: ", train_docs_tf.shape)

    train_docs = docs[:max_size]

    vectorizer = TfidfVectorizer()
    train_docs_tf = vectorizer.fit_transform(train_docs)
    terms = vectorizer.get_feature_names()

    conversation_end = df['conversation_end']
    encoder = LabelEncoder()
    encoder.fit(conversation_end)
    encoded_Y = encoder.transform(conversation_end)
    train_Y = encoded_Y[:max_size]
    test_Y = encoded_Y[max_size:]

    # test_docs = count_vect.transform(docs[max_size:])
    # test_docs_tf = tfidf_transformer.transform(test_docs)
    test_docs = docs[max_size:]
    test_docs_tf = vectorizer.transform(test_docs)

    print("test_docs_tf: ", test_docs_tf.shape)

    # print(train_docs_tf)
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf = clf.fit(train_docs_tf, train_Y)
    predictions = clf.predict(test_docs_tf)
    # predictions = clf.predict(test_docs_tf.toarray())
    print(predictions)

    # accuracy = accuracy_score(test_Y, predictions)
    # print("Accuracy: ", accuracy)

    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=terms, class_names=['No', 'Yes'], filled=True, rounded=True)
    graph = graphviz.Source(dot_data)
    graph.view()

    accuracy = accuracy_score(test_Y, predictions)
    print("Accuracy: ", accuracy)
    print(confusion_matrix(test_Y, predictions))


