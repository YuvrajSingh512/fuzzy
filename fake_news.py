import pandas as pd
df = pd.read_csv('train.csv')
X=df.drop('label',axis=1)
df = df.dropna()
messages = df.copy()
messages.reset_index(inplace=True)

# cleaning the messages title by removing special chars, stopwords and stemming
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,ngram_range=(1,3))
X = cv.fit_transform(corpus).toarray()
y = messages['label']

#Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, y_train)
predictions = logisticRegr.predict(X_test)

score = logisticRegr.score(X_test, y_test)
print("Accuracy: ", end = "")
print(float(score*100))



from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
cfm = metrics.confusion_matrix(y_test, predictions)
labels = ["Fake News","Real News"]
sns.heatmap(cfm, annot=True, fmt="d", linewidths=.5, square = True, cmap = 'Blues_r',xticklabels=labels, yticklabels=labels);
plt.ylabel('Actual label');
plt.xlabel('Predicted label');

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
