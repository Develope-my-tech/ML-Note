import pandas as pd
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import sklearn.model_selection

path = 'mushroom.csv'
mr = pd.read_csv(path, header=None)
label1 = [[0, 1], [1, 0]]
label = []
data = []

for idx, row in mr.iterrows():
    la = row[0]
    label.append(ord(la))
    chs = []
    for ch in row[1:]:
        chs.append(ord(ch))
    data.append(chs)

train_data, test_data, train_label, test_label = train_test_split(data, label)

clf = RandomForestClassifier()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)

score = metrics.accuracy_score(test_label, pre)
print(score)

score2 = sklearn.model_selection.cross_val_score(clf, data, label, cv=5)
# 교차 검증 (대상, 데이터, 정답, 그룹수)
print('cross : ', score2)