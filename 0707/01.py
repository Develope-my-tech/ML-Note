import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

csv = pd.read_csv('./iris/iris.csv')


csv_data = csv[['SepalLength','SepalWidth','PetalLength','PetalWidth']]
# print(csv_data[['SepalLength']])
csv_label = csv['Name']
print(csv_data)
print(csv_label)

train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

score = metrics.accuracy_score(test_label, pre)
print('score', score)