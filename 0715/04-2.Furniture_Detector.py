# 1. 학습모델 생성
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',
                       input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(5,activation='softmax')) # 1일 확률값을 반환
model.summary()
"""
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               3211776   
_________________________________________________________________
dense_1 (Dense)              (None, 5)                 2565      
=================================================================
Total params: 3,455,173
Trainable params: 3,455,173
Non-trainable params: 0
"""



# 2. 컴파일
from keras import optimizers
model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])



# 3. 이미지 전처리
from keras.preprocessing.image import ImageDataGenerator
dir = './img/'
types = ['train/','val/']

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory( #학습 이미지 전처리
   # 타깃 디렉터리
   dir+types[0],
   # 모든 이미지를 150x150 크기로 변경
   target_size=(150,150),
   batch_size = 20,
   class_mode = 'categorical')
validation_generator = test_datagen.flow_from_directory( # 테스팅 이미지 전처리
   dir+types[1],
   target_size = (150,150),
   batch_size = 20,
   class_mode = 'categorical')
"""
Found 4024 images belonging to 5 classes.
Found 423 images belonging to 5 classes.
"""



# 4. 클래스 분류 확인
print(validation_generator.class_indices)
"""
{'bed': 0, 'chair': 1, 'sofa': 2, 'swivelchair': 3, 'table': 4}
"""



# 5.
for data_batch, labels_batch in train_generator:
   print('배치 데이터 크기:', data_batch.shape)
   print('배치 레이블 크기:', labels_batch.shape)
   break
"""
배치 데이터 크기: (20, 150, 150, 3)
배치 레이블 크기: (20, 5)
"""



# 6.학습
history = model.fit_generator(
   train_generator,
   steps_per_epoch = 40, # 배치 개수
   epochs = 30,
   validation_data = validation_generator,
   validation_steps = 5) # 배치 개수




# 7. 테스트
import numpy as np
# {'bed': 0, 'chair': 1, 'sofa': 2, 'swivelchair': 3}
pre = model.predict_generator(validation_generator,steps=1)
for p in pre:
   print(np.argmax(p),end=',')
"""
2,1,3,0,0,1,1,0,2,0,0,3,1,2,0,0,1,3,1,0,
"""

# 8. 정답 확인
from keras.preprocessing import image
import matplotlib.pyplot as plt
i = 0
for i in range(20):
   plt.figure(i)
   imgplot = plt.imshow(validation_generator[0][0][i])
   x = validation_generator[0][0][i]
   x = x.reshape(1, 150,150,3)
   pre = model.predict(x)
   print('예측:',np.argmax(pre),end='/')
   print('정답:',validation_generator[0][1][i])
plt.show() # 그림 띄우기
"""
예측: 2/정답: [0. 0. 1. 0.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 1/정답: [0. 1. 0. 0.]
예측: 1/정답: [0. 1. 0. 0.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 0/정답: [0. 0. 0. 1.]
예측: 3/정답: [0. 0. 0. 1.]
예측: 3/정답: [0. 0. 0. 1.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 0/정답: [1. 0. 0. 0.]
예측: 3/정답: [0. 0. 0. 1.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 2/정답: [0. 0. 1. 0.]
예측: 1/정답: [0. 1. 0. 0.]
"""

# 9. 내 이미지로 테스팅
# {'bed': 0, 'chair': 1, 'sofa': 2, 'swivelchair': 3}
from keras.preprocessing import image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

# a)이미지 로드
img_path = 'C:/Users/Playdata/Downloads/chair.jpeg'
img = image.load_img(img_path,target_size = (150,150))
# b)이미지 행렬 형태로 변환
x = image.img_to_array(img)
print('기존 shape:',x.shape)
# c)예측 가능한 형태로 reshape (4차원 형태)
x = x.reshape((1,)+x.shape) # [1,150,150,3] : 넣어줄 데이터의 개수 + 기존 shape
print('변형한 shape:',x.shape)
# d)예측 값 확인
pre = model.predict(x)
print('예측 값:',np.argmax(pre))
# e)테스팅 이미지 출력
img = mpimg.imread(img_path)
imgplot = plt.imshow(img)
plt.show()
"""
기존 shape: (150, 150, 3)
변형한 shape: (1, 150, 150, 3)
예측 값: 1
"""

# 10. 성능 확인
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
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
