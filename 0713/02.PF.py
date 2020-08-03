from keras import models
from keras import layers
from keras import optimizers
import numpy as np



# 1단계
passing_score = 70
train_x = np.zeros((100,),dtype=np.float32) ##모양
train_y = np.zeros((100,),dtype=np.float32)
val_x = np.zeros((50,),dtype=np.float32)    # test data
val_y = np.zeros((50,),dtype=np.float32)

x1 = np.random.randint(1,101,(100),dtype=np.int32) #1부터 100까지 100개
x2 = np.random.randint(1,101,(50),dtype=np.int32)

for i in range(len(x1)):
   train_x[i]=x1[i]
   if x1[i]>=passing_score:
       res = 1
   else:
       res = 0
   train_y[i]=res

for i in range(len(x2)):
   val_x[i]=x2[i]
   if x2[i]>=passing_score:
       res = 1
   else:
       res = 0
   val_y[i]=res

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal',input_shape=(1,)))
model.add(layers.Dense(16,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer = 'rmsprop',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])

# 모델을 학습시키기위해 fit() 함수를 사용. 학습 이력 정보 리턴
history = model.fit(train_x,
                  train_y,
                  epochs=40,    # 40번씩 반복
                  batch_size=4, # 4개를 하나의 그룹으로
                  validation_data = (val_x,val_y))

# 2단계
his = history.history   # 학습 이력 정보
k = his.keys()
print(k)
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 3단계
pre = model.predict(val_x)
print(pre.shape)    # (50, 1)
print(pre[0])       # [0.69234324]
for i in range(len(pre)):
   if pre[i]>=0.5:
       print('합격')
   else:
       print('불합격')

# 4단계
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,loss,'bo',label='Training loss')    # 'bo' : 파란색 점
plt.plot(epochs,val_loss,'b',label='Validation loss')   # 'b' : 파란색 실선
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
