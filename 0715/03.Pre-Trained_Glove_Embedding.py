# STEP 1
import os

imdb_dir = './aclimdb'
train_dir = os.path.join(imdb_dir,'train')

labels = []
texts = []

for label_type in ['neg','pos']:
   dir_name = os.path.join(train_dir,label_type)
   for fname in os.listdir(dir_name):
       if fname[-4:] == '.txt':
           f = open(os.path.join(dir_name,fname),encoding = 'utf8')
           texts.append(f.read())
           f.close()
           if label_type == 'neg':
               labels.append(0)
           else:
               labels.append(1)

# for l, t in zip(labels, texts):
#     print(l, t)
# l : 0은 부정, 1은 긍정 / t : 문장 (리뷰)



# STEP 2
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

maxlen = 100 # 한 문장에 최대 단어 개수
training_samples = 10000 # 훈련 샘플은 200개*****
validation_samples = 10000 # 검증 샘플은 10,000개
max_words = 10000 # 데이터 셋에서 가장 빈도 높은 10000개의 단어만 사용

tokenizer = Tokenizer(num_words = max_words)
tokenizer.fit_on_texts(texts) # 딕셔너리 생성
sequences = tokenizer.texts_to_sequences(texts) # 모든 문장 인덱싱 ex. [2,5,32,...]

word_index = tokenizer.word_index # 인덱싱 처리한 단어 개수
print('%s개의 고유한 토큰을 찾았습니다.'%len(word_index))

# 패딩: 단어의 개수 100개가 안되는 경우 나머지를 0으로 채우도록 처리
data = pad_sequences(sequences, maxlen=maxlen)
labels = np.asarray(labels)
print('데이터 텐서의 크기:', data.shape)
print('레이블 텐서의 크기:', labels.shape)

# 데이터를 훈련 세트와 검증 세트로 분할
# 샘플이 순서대로 있기 때문에(부정 샘플이 모두 나온 후에 긍정 샘플이 온다.)
# 먼저 데이터를 섞음
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

glove_dir = './aclImdb/'



# STEP 3. glove.6B.100d.txt에 있는 모든 임베딩 벡터들 불러오기 (dict형)
embeddings_index = {}
f = open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding = 'utf8')
for line in f:
   values = line.split()
   word = values[0]
   coefs = np.asarray(values[1:],dtype='float32')
   embeddings_index[word] = coefs
f.close()

print('%s개의 단어 벡터를 찾았습니다.' % len(embeddings_index))
# print(len(embeddings_index['respectable']))   # 100


# STEP 4
embedding_dim = 100
embedding_matrix = np.zeros((max_words,embedding_dim))
# 단어 집합 크기의 행과 100개의 열을 가지는 행렬 생성.

for word, i in word_index.items():  # 훈련 데이터의 단어 집합에서 단어를 1개씩 꺼내옴.
   embedding_vector = embeddings_index.get(word)    # 단어(key) 해당되는 임베딩 벡터의 100개의 값을 임시 변수에 저장.

   if i<max_words:
       if embedding_vector is not None:
           # 임베딩 인덱스에 없는 단어는 모두 0이 된다.
           embedding_matrix[i] = embedding_vector



# STEP 5. 임베딩 층(embedding layer) 만들기
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()



# STEP 6
model.layers[0].set_weights([embedding_matrix]) # 미리 학습한 내용을 가중치로 설정
# 모델 동결. 새 학습을 통해 나온 데이터로 갱신하는 것을 막음
model.layers[0].trainable = False



# STEP 7
model.compile(optimizer = 'rmsprop',
            loss = 'binary_crossentropy',
            metrics = ['acc'])
history = model.fit(x_train,y_train,
                  epochs = 10,
                  batch_size = 32,
                  validation_data = (x_val,y_val))
model.save_weights('pre_trained_glove_model.h5')



# STEP 8. 학습 결과 확인
x_test = x_val[:10]
y_test = y_val[:10]
print(y_test)
pre = model.predict(x_test)
print(pre)



# STEP 9. 그래프를 통한 성능 확인
# 그래프로 성능 확인 => 각 단어 사이의 관계를 살펴 봄 (정확도 좋지 않다)
# => 필터링 된 결과를 누적하여 문장을 해석해야 정확한 내용 파악 가능.
# 즉, dense가 더 올바름.
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
