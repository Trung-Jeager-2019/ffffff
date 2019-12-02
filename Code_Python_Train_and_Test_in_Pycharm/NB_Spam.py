# Read me
'''
Cài đặt thư viện - Câu lệnh commandline với terminal, powershell, ...

pip install pandas
pip install scikit-learn
pip install numpy

'''

import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

messages = pd.read_csv("Data_set/spam.csv", encoding='utf-8', engine='python')
# Drop the extra columns and create columns
messages = messages.drop(labels=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
# Create 2 name column (category, text)
messages.columns = ["category", "text"]
# Take dataset category
messages['category'] = messages['category'].map({'ham': 0, 'spam': 1})
# Take dataset text
category = messages['category']
text = messages['text']
# Extract Feature With CountVectorizer
cv = CountVectorizer()
# Fit the Data
text = cv.fit_transform(text)
# import train_test_split
X_train, X_test, y_train, y_test = train_test_split(text, category, test_size=0.33, random_state=42)
# import MultinomialNB - Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)
# Build model with train - Alternative Usage of Saved Model
joblib.dump(clf, 'Data_set/NB_spam_model.pkl')

NB_spam_model = open('Data_set/NB_spam_model.pkl', 'rb')
clf = joblib.load(NB_spam_model)
clf.score(X_test, y_test)

input_mesages = input("Nhập văn bản email cần kiểm tra tại đây: ")

data = [input_mesages]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)

if my_prediction == 1:
    print("Đây là văn bản email Spam!")
else:
    print("Đây không phải là email Spam!")

