import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
messages = pd.read_csv('/Users/ashwariyasah/Projects/nlp/sms+spam+collection/SMSSpamCollection', sep='\t',
names=['label', 'message'])

# print(messages.head())

corpus = []
wnl = WordNetLemmatizer()
for i in range(0, len(messages)):
    text = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    text = text.lower()
    text = text.split()
    text = [wnl.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = ' '.join(text)
    corpus.append(text)

#print(corpus)

#bag of words
from sklearn.feature_extraction.text import CountVectorizer
# cv = CountVectorizer(max_features=2500)
# X = cv.fit_transform(corpus)

# Create TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2500)
X = vectorizer.fit_transform(corpus)

# don't really need toarray()

print(X.shape)


y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

#traim test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

# print(X_train.shape)
# print(y_train.shape)
from sklearn.naive_bayes import MultinomialNB
# Create the Multinomial Naive Bayes classifier
spam_detect_model = MultinomialNB()
spam_detect_model.fit(X_train, y_train)
y_pred = spam_detect_model.predict(X_test)

#print(y_pred.shape

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy is", accuracy)
