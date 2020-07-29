import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV

path = '../0707/mnist/'


def load_csv(fname):
    label = []
    data = []
    with open(path + fname, 'r') as f:
        for line in f:
            t = line.split(',')
            if t[0] == '\n':
                continue
            label.append(int(t[0]))
            num_img = []  # 숫자 이미지 1개
            for i in range(1, len(t)):
                num_img.append(int(t[i]) / 256)
            data.append(num_img)
    return label, data


train_label, train_data = load_csv('train.csv')
test_label, test_data = load_csv('t10k.csv')


params = [{"C": [1, 10, 100, 1000], "kernel": ["linear"]},
          {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]}]
clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)
clf.fit(train_data, train_label)
pre = clf.predict(test_data)
print(pre)
score = metrics.accuracy_score(test_label, pre)
print(score)