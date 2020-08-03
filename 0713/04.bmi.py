# empty: 쓰레기값 들어있는 벡터 생성
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping
import pandas as pd, numpy as np

csv = pd.read_csv('bmi.csv')

# 데이터로 사용할 값 소수점으로 만들기
csv['weight'] /= 100
csv['height'] /= 200

# 학습 데이터
X = csv[['weight','height']] # keras에서 사용할 수 있는 벡터로 변환

# y. one_hot encoding
bclass = {'thin':[1,0,0],'normal':[0,1,0],'fat':[0,0,1]}

# width3 20000줄인 쓰레기값으로 초기화된 벡터
y = np.empty((20000,3))

# y로 생성
for i,v in enumerate(csv['label']):
   y[i]=bclass[v]

# train용과 검증용 데이터 분리
X_train,y_train = X[1:15001],y[1:15001]
X_test,y_test = X[15001:20001],y[15001:20001]

# 모델 생성 및 구성
model = Sequential()
model.add(Dense(512,input_shape=(2,)))
model.add(Activation('relu'))
model.add(Dropout(0.1)) # 과적합을 막기 위해 버리기 데이터의 10퍼센트 누락시킴(관계에서 규칙성이 생겨 오류 범할 수 있기 때문에. 일부러 주변의 몇개 데이터 누락시킴)
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(Dense(3))
model.add(Activation('softmax')) # 여러 가지 case를 다루기 위해서. 각 결과의 %를 계산

model.compile(
   loss = 'categorical_crossentropy',
   optimizer = 'rmsprop',
   metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 4,
                  batch_size = 512,
                  validation_data = (X_test,y_test))

score = model.evaluate(X_test,y_test)
print('loss=',score[0])
print('accuracy=',score[1])
