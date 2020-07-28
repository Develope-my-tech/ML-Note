import csv, cv2
import pandas as pd
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split

# 알파벳 사진 잘라내기
gray_img = cv2.imread('alph/alphabets.png', 0)
al = 'abcdefghijklmnopqrstuvwxyz'
h, w = gray_img.shape
al_w = w/26
ww = 0
imgs = []

for i in range(0, 26):
    img = gray_img[0:h, int(i*al_w):int(ww+al_w)]
    ww += al_w
    imgs.append(img)

# 알파벳 바이너리 파일을 csv파일로 만들어서 저장하기.
# idx = 0
# csv_f = open('alph/al.csv', 'w', encoding='utf-8')
# for x in imgs:
#     fdata = x.ravel()
#     img_data = list(map(lambda n: str(n), fdata))
#     csv_f.write(al[idx] + ',')
#     csv_f.write(','.join(img_data) + '\r\n')
#     idx+=1

path = 'alph/'
train_label = []
train_data = []

with open(path+'al.csv', 'r') as f:
    for line in f:
        if len(line)==1:
            continue
        l = line.strip().split(',')
        train_label.append(l[0])
        train_data.append([int(a) for a in l[1:]])

# train_data, test_data, train_label, test_label = train_test_split(train_data, train_label)

clf = svm.SVC()
clf.fit(train_data, train_label)
# t = cv2.imread('alph/test1.PNG', 0)
# t = t.reshape(1,  h, int(w/26))
pre = clf.predict([train_data[0]])
# score = metrics.accuracy_score(test_label, pre)
print(pre)
# print('score:', score)