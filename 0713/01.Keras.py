from keras import models
from keras import layers
from keras import optimizers

# Dense Layer
# https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/

# 학습데이터
train_data = [[0,0],[0,1],[1,0],[1,1]]
train_label = [[0],[1],[1],[0]]
# 테스팅 데이터
test_data =  [[1,0],[1,1],[0,0],[0,1]]
test_label = [[1],[0],[0],[1]]

# 학습 모델 생성
model = models.Sequential()
#kernel_initializer: 가중치 초기화
#bias_initializer: 바이어스 초기화
model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal',input_shape=(2,)))
# model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal',input_dim=2)) 과 같음.
model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal'))
model.add(layers.Dense(1,activation='sigmoid'))
#relu: 음수를 0으로 수렴시킴
#sigmoid: 0에서 1로 수렴
#softmax: 다양한 결과값에 수렴.예측되는 각 결과값의 확률을 반환

model.compile(optimizer='rmsprop',
             loss = 'binary_crossentropy', #0,1
             metrics=['accuracy'])
#학습
history = model.fit(train_data, train_label,
                  epochs = 20,
                  batch_size=4,
                  validation_data = (test_data,test_label))

# 2단계
his = history.history
k = his.keys()
print(k)

# 3단계
print(his['loss'],'/',his['accuracy'])

# 4단계
pre = model.predict(test_data)
print(pre.shape)
print(pre[0])
for i in range(len(pre)):
   if pre[i]>=0.5: ##
       print(1)
   else:
       print(0)

# 5단계
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

#'bo'는 파란색 점을 의미합니다.
plt.plot(epochs,loss,'bo',label='Training loss')
# 'b'는 파란색 실선을 의미합니다.
plt.plot(epochs, val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs') # 반복 횟수
plt.ylabel('Loss')
plt.legend()

plt.show()