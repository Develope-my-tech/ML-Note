# <Version1. Dense>
# 1. 기사 로드
from keras.datasets import reuters
(train_data,train_labels), (test_data,test_labels)= reuters.load_data(num_words=10000)


# 2. 문자로 제대로 출력되는지 테스트
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])
# 0,1,2는 각각 '패당','문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺀다.
decoded_newswire = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decoded_newswire) #확인
'''
'? ? ? said as a result of its december acquisition of space
co it expects earnings per share in 1987 of 1 15 to 1 30 dlrs
 per share up from 70 cts in 1986 the company said pretax net
  should rise to nine to 10 mln dlrs from six mln dlrs in 1986
   and rental operation revenues to 19 to 22 mln dlrs from 12 5
    mln dlrs it said cash flow per share this year should be 2 50
     to three dlrs reuter 3'
'''


# 3. 데이터 벡터화: 단어가 있는 부분 1 없는 부분 0인 벡터 생성해서 리턴
import numpy as np

def vectorize_sequences(sequences,dimension = 10000):
   results = np.zeros((len(sequences),dimension))
   for i, sequence in enumerate(sequences):
       # 2차원 행렬에서 i번째 줄의 sequence에 담긴 숫자를 인덱스로 하여 그 위치를 1로 세팅
       results[i,sequence] =1.
   return results
# 훈련 데이터 벡터 변환
x_train = vectorize_sequences(train_data)
# 테스트 데이터 벡터 변환
x_test = vectorize_sequences(test_data)



# 4-1. 레이블 one-hot encoding => 46칸의 벡터로 변환
def to_one_hot(labels,dimension = 46):
   results = np.zeros((len(labels),dimension))
   for i,label in enumerate(labels):
       results[i,label] = 1.
   return results
# 훈련 레이블 벡터 변환
one_hot_train_labels = to_one_hot(train_labels)
# 테스트 레이블 벡터 변환
one_hot_test_labels = to_one_hot(test_labels)

# 4-2. to_categorical API 사용 one-hot encoding
from keras.utils.np_utils import to_categorical
# 훈련 레이블 벡터 변환
one_hot_train_labels = to_categorical(train_labels)
# 테스트 레이블 벡터 변환
one_hot_test_labels = to_categorical(test_labels)



# 5. dense를 이용한 처리(Embedding보다 정확도 높음)
from keras import models
from keras import layers
model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax')) # 카테고리 개수 46개이기 때문에.



# 6. 컴파일
model.compile(optimizer='rmsprop',
            loss = 'categorical_crossentropy',# binary가 아닌 categorical 손실함수 사용
            metrics = ['accuracy'])



# 7. 데이터 분배.(train, validation 분리)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]



# 8. 학습
history = model.fit(partial_x_train,
                  partial_y_train,
                  epochs= 9,
                  batch_size = 512,
                  validation_data = (x_val,y_val))



# 9. 예측 테스트
sample = x_test[:10]
pre = model.predict(sample)
for p in pre:
   print(np.argmax(p),end=',') # 기사별 확률 가장 높은 토픽 idx 출력
   # 3,10,1,4,13,3,3,3,3,3,

# 비교
print(test_labels[:10])
# [ 3 10  1  4  4  3  3  3  3  3]


# <version2. Embedding>
# 1. 기사로드~ 3. 데이터 패딩(벡터화?)
# from keras.datasets import reuters
# (train_data,train_labels), (test_data,test_labels)= reuters.load_data(num_words=10000)
# x_train=preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
# x_test=preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)
#
# # 4. to_categorical API 사용 one-hot encoding
# from keras.utils.np_utils import to_categorical
# # 훈련 레이블 벡터 변환
# one_hot_train_labels = to_categorical(train_labels)
# # 테스트 레이블 벡터 변환
# one_hot_test_labels = to_categorical(test_labels)
#
# print(one_hot_train_labels.shape)
# '''
# (8982, 46)
# '''
#
# # 5. Embedding를 이용한 처리(Dense보다 정확도 낮음)
# model = models.Sequential()
# model.add(Embedding(8982,100,input_length=maxlen)) # one_hot_train_labels.shape: (8982, 46)
# model.add(Flatten())
# model.add(Dense(46,activation='softmax'))
#
# # 6. 컴파일
# model.compile(optimizer='rmsprop',
#             loss = 'categorical_crossentropy',# binary가 아닌 categorical 손실함수 사용
#             metrics = ['acc'])
# model.summary()
# '''
# Model: "sequential_5"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_2 (Embedding)      (None, 100, 100)          898200
# _________________________________________________________________
# flatten_2 (Flatten)          (None, 10000)             0
# _________________________________________________________________
# dense_9 (Dense)              (None, 46)                460046
# =================================================================
# Total params: 1,358,246
# Trainable params: 1,358,246
# Non-trainable params: 0
# _________________________________________________________________
# '''
#
# # 7. 데이터 분배.(train, validation 분리)
# x_val = x_train[:1000]
# partial_x_train = x_train[1000:]
# y_val = one_hot_train_labels[:1000]
# partial_y_train = one_hot_train_labels[1000:]
#
# # 8. 학습
# history = model.fit(partial_x_train,
#                   partial_y_train,
#                   epochs= 9,
#                   batch_size = 512,
#                   validation_data = (x_val,y_val))
