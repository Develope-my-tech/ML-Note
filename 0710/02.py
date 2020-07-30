import tensorflow.compat.v1 as tf
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.compat.v1.disable_eager_execution()  # 옛날 버전으로 쓸 수 있게 도와주는 함수.

a = tf.placeholder(tf.int32, [3])
b = tf.constant(2)
x2_op = a * b

sess = tf.Session()
r1 = sess.run(x2_op, feed_dict={ a:[1, 2, 3] })
print(r1)
r2 = sess.run(x2_op, feed_dict={ a:[10, 20, 10] })
print(r2)