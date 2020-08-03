import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import pickle

csv = pd.read_csv('heart.csv')
#값의 범위를 0~1로 만들어줌
csv['age']=csv['age']/100
csv['trestbps']=csv['trestbps']/200
csv['chol']=csv['chol']/500
csv['thalach']=csv['thalach']/100
csv_data = csv[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
csv_label = csv["target"]
# print(csv_data)
# print(csv_label)

train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pre)
# print(pre)
print('score:', score)  # score: 0.8289473684210527

with open('heart_res.pkl', 'wb') as f:
    pickle.dump(clf, f)#오픈한 파일에 저장

with open('heart_res.pkl', 'rb') as f:
    clf = pickle.load(f)

your_data = [0.64,1,3,1,0,0,0,1,1,1.8,1,0,2] # pre:1

pre = clf.predict([your_data])
print(pre) # [1]