import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf

tf.compat.v1.disable_eager_execution()

# 키, 몸무게, 레이블이 적힌 CSV 파일 읽어 들이기 --- (※1)
csv = pd.read_csv("iris.csv")

# 데이터 정규화 --- (※2)
csv["SepalLength"] = csv["SepalLength"]/10
csv["SepalWidth"] = csv["SepalWidth"]/10
csv["PetalLength"] = csv["PetalLength"]/10
csv["PetalWidth"] = csv["PetalWidth"]/10
# one-hot 레이블

# 레이블을 배열로 변환하기 --- (※3)
# - thin=(1,0,0) / normal=(0,1,0) / fat=(0,0,1)
bclass = {"Iris-setosa": [1,0,0], "Iris-versicolor": [0,1,0], "Iris-virginica": [0,0,1]}
csv["label_name"] = csv["Name"].apply(lambda x : np.array(bclass[x]))

# 테스트를 위한 데이터 분류 --- (※4)
test_csv = csv[50:150]
test_pat = test_csv[["SepalLength","SepalWidth","PetalLength","PetalWidth"]]
test_ans = list(test_csv["label_name"])

# 데이터 플로우 그래프 구축하기 --- (※5)
# 플레이스홀더 선언하기
x  = tf.placeholder(tf.float32, [None, 4]) # 키와 몸무게 데이터 넣기
y_ = tf.placeholder(tf.float32, [None, 3]) # 정답 레이블 넣기

# 변수 선언하기 --- (※6)
W = tf.Variable(tf.zeros([4, 3])) # 가중치
b = tf.Variable(tf.zeros([3])) # 바이어스
# 실제 값과 예측 값의 분산을 계산할 회귀 함수 지정.

# 소프트맥스 회귀 정의하기 --- (※7)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 모델 훈련하기 --- (※8)
# 손실을 최소화 하기 위한 최적화 코드
cross_entropy = -tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(cross_entropy)

# 정답률 구하기
predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

# 세션 시작하기
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # 변수 초기화하기

# 학습시키기
for step in range(150):
    data = csv[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]
    ans = list(csv["label_name"])
    fd = {x: data, y_: ans}
    sess.run(train, feed_dict=fd)
    if step % 10 == 0:
        cre = sess.run(cross_entropy, feed_dict=fd)
        acc = sess.run(accuracy, feed_dict=fd)
        print("step=", step, "cre=", cre, "acc=", acc)

# 최종적인 정답률 구하기
acc = sess.run(accuracy, feed_dict=fd)
print("정답률 =", acc)
pre = sess.run(tf.argmax(y, 1), feed_dict=fd)
#print(pre)
for idx in pre:
    print(list(bclass.values())[idx])