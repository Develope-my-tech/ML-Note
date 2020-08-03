import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()

xor_data = [[0, 0],[0, 1],[1, 0],[1, 1]]
xor_label = [[0],[1],[1],[0]]

x  = tf.placeholder(tf.float32, [None, 2]) # 데이터
y_ = tf.placeholder(tf.float32, (None, 1)) # 레이블

W = tf.Variable(tf.zeros([2, 1]))  # 가중치, 데이터 폭, 레이블 폭
b = tf.Variable(tf.zeros([1]))  # 바이어스. 레이블 폭

# 소프트맥스 회귀함수 정의
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 손실률 최소화
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 계산 수식
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

#실행
sess = tf.Session()
sess.run(tf.global_variables_initializer()) #변수 초기화
fd = {x: xor_data, y_: xor_label}

sess.run(train, feed_dict=fd)   # 훈련
cre = sess.run(cross_entropy, feed_dict=fd)  # 가중치와 바이오스 재조정.
acc = sess.run(accuracy, feed_dict={x: xor_data, y_: xor_label}) # 테스팅
print("정답률 =", acc)

pre = sess.run(tf.argmax(y, 1), feed_dict={x: xor_data})    # 학습 결과 확인
print(pre)