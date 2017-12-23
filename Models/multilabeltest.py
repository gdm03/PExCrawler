import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

df = pd.read_csv('testmlb.csv')

# X_train = np.array(["new york is a hell of a town",
#                     "new york was originally dutch",
#                     "the big apple is great",
#                     "new york is also called the big apple",
#                     "nyc is nice",
#                     "people abbreviate new york city as nyc",
#                     "the capital of great britain is london",
#                     "london is in the uk",
#                     "london is in england",
#                     "london is in great britain",
#                     "it rains a lot in london",
#                     "london hosts the british museum",
#                     "new york is great and so is london",
#                     "i like london better than new york"])

# y_train_text = [["new york"],["new york"],["new york"],["new york"],["new york"],
#                 ["new york"],["london"],["london"],["london"],["london"],
#                 ["london"],["london"],["new york","london"],["new york","london"]]
# print(y_train_text)
#
# print("-----------------------------------------------------------------------------")

X_train = df['post_content'].dropna().values.astype('U')

#test = df['dialogue_acts'].dropna().values.astype('U').tolist()
y_train_text = df['dialogue_acts'].str.split(', ').dropna().tolist()
#y_train_text = [[x] for x in test]
print(y_train_text)

# X_test = np.array(['nice day in nyc',
#                    'welcome to london',
#                    'london is rainy',
#                    'it is raining in britian',
#                    'it is raining in britian and the big apple',
#                    'it is raining in britian and nyc',
#                    'hello welcome to new york. enjoy it here and london too'])

X_test = np.array(['hey?',
                   'Not sure what I should say.',
                   'london is rainy',
                   'it is raining in britian',
                   'it is raining in britian and the big apple',
                   'it is raining in britian and nyc',
                   'hello welcome to new york. enjoy it here and london too'])
#target_names = ['New York', 'London']
target_names = ['Comment', 'Question', 'Backchannel', 'Expression', 'Others']

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train_text)
print(Y)
print(list(mlb.classes_))

classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])

classifier.fit(X_train, Y)
predicted = classifier.predict(X_test)
all_labels = mlb.inverse_transform(predicted)

for item, labels in zip(X_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))
