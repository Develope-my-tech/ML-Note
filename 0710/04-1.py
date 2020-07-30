import pandas as pd
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

xor_data = [[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]]
xor_label = [[0.],
            [1.],
            [1.],
            [0.]]

x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 2])
y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1])
W = tf.compat.v1.Variable(tf.compat.v1.random.normal([2, 4]))
b = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))

xh1 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(x, W) + b)  # 여기까지

Wh1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([4, 1]))
bh1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))

y = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(xh1, Wh1) + bh1)  # 여기까지
# y = tf.compat.v1.math.round(y)
cross_entropy = -tf.compat.v1.reduce_sum((y_ * tf.compat.v1.log(y)) + ((1 - y_) * tf.compat.v1.log(1 - y)))
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)  # AdamOptimizer(learning_rate=0.01,)#
train = optimizer.minimize(cross_entropy)

test = y
predict = tf.compat.v1.equal(tf.compat.v1.math.round(y), y_)
accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(predict, tf.compat.v1.float32))

sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())
fd = {x: xor_data, y_: xor_label}
for i in range(10000):
   sess.run(train, feed_dict=fd)

pre = sess.run(predict, feed_dict={x: xor_data, y_: xor_label})
print(pre)
acc = sess.run(accuracy, feed_dict={x: xor_data, y_: xor_label})
print("정답률=", acc)
pre = sess.run(tf.compat.v1.math.round(y), feed_dict={x: [[0, 1]]})
print(pre)
check_y = sess.run(y, feed_dict={x: xor_data})
print("y:", check_y)
# 와 했다
# xor 를 가능하게 하기위해 필요한 조건
# 1. variable 의 random.normal
# 2. 3층 이상
# 3. 3층이라면 히든 layer 의 노드의 개수가 4개 이상