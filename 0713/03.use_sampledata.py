from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers

def vectorize_sequences(sequences,dimension=10000): # 문장, 폭=10000
   # 크기가 (len(sequences),dimension))이고 모든 원소가 0인 행렬을 만듭니다.
   results = np.zeros((len(sequences),dimension)) #10000칸짜리 배열 초기화.행렬
   for i, sequence in enumerate(sequences):
       results[i,sequence] = 1. #results[i]에서 특정 인덱스의 위치를 1로 만듭니다
   return results

(train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)
# print(len(train_data), len(test_data))  # 25000, 25000
print(train_data[0]) # 0번째 리뷰 확인
# [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66 ... ] ( 리뷰에서 자주 나오는 단어들의 인덱스 )
# 훈련 데이터를 벡터로 변환
x_train = vectorize_sequences(train_data)
# 테스트 데이터를 벡터로 변환
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=0.001), #optimizer에서 사용하는 gredient 상수를 0.001로 지정
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

# 데이터 분류 (위의 테스트 데이터 사용해도 됨)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,
                  partial_y_train,
                  epochs = 20,
                  batch_size = 512,
                  validation_data = (x_val,y_val))

history_dict = history.history
history_dict.keys()


# 손실율 그래프
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 정확도 그래프
plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()



# test_data(리뷰 리스트)의 리뷰 10개를 추출(0번째 방부터) 문장으로 변환
# 이를 predict하여 긍정 문장인지 부정 문장인지 출력

# print(train_data[0])
word_index = imdb.get_word_index()
# 정수 인덱스와 단어를 매핑하도록 뒤집습니다.
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])

# 리뷰를 디코딩합니다.
# 0,1,2는 '패딩','문서 시작','사전에 없음'을 위한 인덱스이므로 3을 뺍니다.
test_sample = x_test[:10]
pre = model.predict(test_sample)
for i in range(10):
   if i>0.5:
       print('긍정 글>>>')
   else:
       print('부정 글>>>')
   decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in test_data[i]])
   print(decoded_review)