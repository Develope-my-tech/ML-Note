from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pandas as pd

csv = pd.read_csv('bmi/bmi.csv')

csv_data = csv[['height', 'weight']]
csv_label = csv['label']

train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

score = metrics.accuracy_score(test_label, pre)
print(score)