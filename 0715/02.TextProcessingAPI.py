# STEP 1
from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

# 가장 빈도가 높은 1000개의 단어만 선택하도록 tokenizer 객체 생성
tokenizer = Tokenizer(num_words=1000) # 자동으로 단어마다 번호를 부여하는 사전 생성.
tokenizer.fit_on_texts(samples)



# STEP 2. 문자열을 정수 인덱스의 리스트로 변환
sequences = tokenizer.texts_to_sequences(samples)
# print(sequences)    # [[1, 2, 3, 4, 1, 5], [1, 6, 7, 8, 9]]

# 직접 one-hot binary vector 표현을 얻을 수 있다.
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
# print(one_hot_results)
# [[0. 1. 1. ... 0. 0. 0.]
#  [0. 1. 0. ... 0. 0. 0.]]
# print(one_hot_results.shape)      # (2, 1000)
# print(one_hot_results[0][3])      # 1.0   사전에서 찾은 경우 단어의 경우 1
# print(one_hot_results[0][10])     # 0.0   사전에서 찾지 못한 단어의 경우 0

# 몇개의 단어를 처리했는 지 개수
word_idx = tokenizer.word_index
print("Found {} unique tokens".format(len(word_idx)))



# STEP 3
from keras.datasets import imdb
from keras import preprocessing
import numpy as np

# 특성으로 사용할 단어의 수
max_features = 10000
# 사용할 텍스트의 길이(가장 빈번한 max_features 개의 단어만 사용합니다)

maxlen = 20 # 한 문장의 최대 단어 수

# 정수리스트로 데이터를 로드
(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)     # (25000,) (25000,) (25000,) (25000,)

# 리스트를 (samples,maxlen) 크기의 2D 정수 텐서로 변환
x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen)
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)



# STEP 4. 모델 생성
from keras.models import Sequential
from keras.layers import Flatten,Dense,Embedding

model = Sequential()

# 나중에 임베딩 된 입력을 Flatten 층에서 펼치기 위해 Embedding 층에 input_length를 지정
# param1: 줄수(단어수)/param2: 필터 폭/ param3: 데이터 폭. 단어 최대 길이
# Embedding 층의 출력 크기는 (samples,maxlen,8) 이 된다
model.add(Embedding(10000,8,input_length=maxlen)) # 임베딩(끼워넣기) 층(데이터 개수, 필터 depth, 단어 최대 길이 폭)
# 첫번째 인자 = 단어 집합의 크기. 즉, 총 단어의 개수
# 두번째 인자 = 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
# input_length = 입력 시퀀스의 길이

# 3D 임베딩 텐서를 (samples, maxlen * 8) 크기의 2D 텐서로 펼침
model.add(Flatten())

# 분류기 추가
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()

# 학습
history = model.fit(x_train,y_train,
                  epochs = 10, # 10번 반복
                  batch_size = 32,# 32마다
                  validation_split = 0.2) # 학습 데이터에서 20퍼 떼어내서 테스팅에 사용해라



# STEP 5. 테스팅
# sigmoid층으로부터 도출한 결과. 0.5보다 작으면 부정. 크면 긍정
pre = model.predict(x_test)
print(pre)


# STEP 6. 성능 확인
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs, val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss,'bo',label='Training loss')
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
