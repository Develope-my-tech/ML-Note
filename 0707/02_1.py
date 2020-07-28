from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
path = 'mnist/'
train_label = []
train_data = []


with open(path+'train.csv', 'r') as f:
    for line in f:
        l = line.split(',')
        if l[0] == '\n':
            continue
        train_label.append(int(l[0]))
        tmp = list(map(int, list(l[1:])))
        train_data.append(tmp)

# with open(path + 'train.csv', 'r') as f:
#     for line in f:
#         t = line.split(',')
#         if t[0] == '\n':
#             continue
#         train_label.append(int(t[0]))
#         num_img = []
#         for i in range(1, len(t)):
#             num_img.append(int(t[i]))
#         train_data.append(num_img)

# print(train_label)
# print(train_data)

train_data, test_data, train_label, test_label = train_test_split(train_data, train_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
pre = clf.predict(test_data)
score = metrics.accuracy_score(test_label, pre)
print(pre)
print('score:', score)