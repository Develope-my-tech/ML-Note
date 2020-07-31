-   (1) 머신러닝 정의 및 지도학습, 비지도학습 차이 
	[https://ellun.tistory.com/103?category=276044](https://ellun.tistory.com/103?category=276044)
-   (2) 지도학습 알고리즘(회귀분석, 의사결정나무, SVM)
    [https://ellun.tistory.com/106](https://ellun.tistory.com/106)
-   (3) 신경망과 딥러닝
	[https://ellun.tistory.com/104?category=276044](https://ellun.tistory.com/104?category=276044)

## Tensorflow
데이터 플로우 그래프를 사용하여 수치 연산을 하는 오픈 소스 소프트웨어 라이브러리.
텐서(데이터)들이 노드(연산)을 거치면서 변형되어 다시 엣지를 통해 흘러가는 과정을 제어하기 위한 머신러닝 툴.


1) 기본 구조
		![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://t1.daumcdn.net/cfile/tistory/2256FA33596D8E7029)

	![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://t1.daumcdn.net/cfile/tistory/24A52033596D904436)
	![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https://t1.daumcdn.net/cfile/tistory/21B8F533596D921B33)

2) 자료형
	a. tf.constant : 상수 자료형
	b. tf.Variable : 기계가 사용하는 변수
	c. tf.Placeholder : 사용자가 사용하는 변수

3) 코딩의 흐름
	a. tensor 값 세팅 ( data setting)
	b. 노드 구조를 구축 ( 연산 구조 setting ) ⇒ 그래프 구조 생성
	c. 세션 생성 ( 위치 변동 가능 )
	d. 결과 실행 run

## 01 & 02
1. tensor 값 세팅

		a = tf.compat.v1.constant(120, name="a")
		b = tf.compat.v1.constant(130, name="b")
		c = tf.compat.v1.constant(140, name="c") # 텐서 세팅(상수)
		v = tf.compat.v1.Variable(0, name="v") #텐서 세팅(변수

2. 노드 구조 구축

		calc_op = a + b + c

3. 세션 새성

		with tf.compat.v1.Session() as sess:

4. 결과 실행

		sess.run(assign_op)

## 03. Tensorflow를 이용한 bmi
- **one-hot encoding** : label 값이 연속적인 숫자로 구성되어 있을 경우, 확률에 영향을 미치게 됨. 
예를 들어 0 : 딸기 / 1 : 사과 / 2 : 배 이라고 할때 
1 + 1  = 2 라는 수식이 곧, 딸기 + 딸기 = 배 라는 결과를 초래할 수 있다는 소리.
따라서 고의로 label 값을 리스트나 문자열의 형태로 변경하여 학습.
- 가중치  W 와 편향 b
	- W : weight, 입력 신호가 결과 출력에 주는 영향도를 조절하는 매개변수
	- b : bias,  뉴런(노드)이 얼마나 쉽게 활성화(1) 되느냐를 조정하는 매개변수

		  W = tf.Variable(tf.zeros([2, 3])) # 가중치  
		  b = tf.Variable(tf.zeros([3])) # 바이어스

- softmax
	입력을 0~1 사이의 값으로 채우고 그 모든 logits의 합이 1이 되도록 ouput을 정규화

		y = tf.nn.softmax(tf.matmul(x, W) + b)
		# logits을 받아서 softmax activation을 리턴
		# 이때 y는 '예측값'
	![enter image description here](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=http://cfile6.uf.tistory.com/image/9936A2385B75492A045026)

- cross entropy
	> $H_p (q)=−\sum_{i=1}^{n}  q(x_i) log{p(x_i)}$

	``cross entropy``는 실제 분포 q에 대하여 알지 못하는 상태에서 **모델링을 통하여 구한 분포인 p를 통하여 q를 예측하는 것**이다.
	cross entropy에서 실제값과 예측값이 맞는 경우 0으로 수렴, 틀릴 경우 값이 커지기 때문에 **실제값과 예측값의 차이를 줄이기 위한 entropy**이다. 

	  cross_entropy = -tf.reduce_sum(y_ * tf.log(y)+(1-y_)*tf.log(1-y))  

	``y_ * tf.log(y)+(1-y_)*tf.log(1-y) ``에는 데이터들의 cross entropy가 2차원 텐서 형태로 담겨져 있다. ``reduce_sum()`` 를 통해  이 cross entropy의 합을 구한다. (지정된 차원을 따라 합을 내는 함수. 2차원인 경우 지정할 차원이 1개밖에 없으므로 차원을 따로 지정하지 않음)

- Gradients, 경사 하강법

	  optimizer = tf.train.GradientDescentOptimizer(0.01)  
	  train = optimizer.minimize(cross_entropy)

	1) 모델 parameter W, b에 대해 loss function의 미분을 구하는 작업을 통해 최적값을 찾는다.
	2) cross entropy를 최소화하는 최적값 W, b를 찾는 optimizer로 train.

- predict / accuracy

	  predict = tf.equal(tf.argmax(y, 1), tf.argmax(y_,1))  
	  accuracy = tf.reduce_mean(tf.cast(predict, tf.float32))

	tf.cast(조건) : 조건에 따라 1 또는 0이 반환된다.

- training

	  x_pat = rows[["weight","height"]]   # 몸무게와 키만 가져옴.  
	  y_ans = list(rows["label_pat"])  
	  fd = {x: x_pat, y_: y_ans}  
	  sess.run(train, feed_dict=fd)

	- feed_dict : 값들에 매핑되는 그래프 요소들의 딕셔너리.

- cross entropy 및 accuracy 확인
	
		cre = sess.run(cross_entropy, feed_dict=fd)  
		acc = sess.run(accuracy, feed_dict={x: test_pat, y_: test_ans})

## 04. xor training

 - xor 를 가능하게 하기위해 필요한 조건  
   1. variable 의 random.normal  ⇒ 만약 초기값이 0으로 같은 값이 주어지면 변경되는 w, b의 값도 유사하게 도출되기 때문에 0 대신 random.normal로 초기화
   2. 3층이라면 히든 layer 의 노드의 개수가 4개 이상 ⇒ 데이터가 많지 않기 때문에 4개부터는 의미 없는 w, b값이 나올 수 있다.

- hidden layer 1개로 구축한 신경망

	  y = tf.nn.softmax(tf.matmul(x, W) + b) # hidden layer

## 04-1. (04.py ver2)
- hidden layer 2개로 축한 신경망 모델 ⇒ 정확도 향상

	  x = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 2])  
	  y_ = tf.compat.v1.placeholder(tf.compat.v1.float32, [None, 1])  
	  W = tf.compat.v1.Variable(tf.compat.v1.random.normal([2, 4]))  
	  b = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))  
	  
	  xh1 = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(x, W) + b)  # 여기까지  
	  
	  Wh1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([4, 1]))  
	  bh1 = tf.compat.v1.Variable(tf.compat.v1.random.normal([1]))  
	  
	  y = tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(xh1, Wh1) + bh1)  # 여기까지

## 05. iris.csv 

	  # 출력
	  step= 0 cre= 280.62003 acc= 0.33333334
	  step= 10 cre= 236.96577 acc= 0.6666667
	  step= 20 cre= 208.07503 acc= 0.6666667
	  step= 30 cre= 188.7417 acc= 0.6933333
	  step= 40 cre= 175.01434 acc= 0.75333333
	  step= 50 cre= 164.66325 acc= 0.81333333
	  step= 60 cre= 156.46301 acc= 0.82
	  step= 70 cre= 149.71269 acc= 0.84
	  step= 80 cre= 143.98979 acc= 0.86
	  step= 90 cre= 139.02649 acc= 0.88666666
	  step= 100 cre= 134.6451 acc= 0.9066667
	  step= 110 cre= 130.72298 acc= 0.9066667
	  step= 120 cre= 127.17258 acc= 0.93333334
	  step= 130 cre= 123.929405 acc= 0.93333334
	  step= 140 cre= 120.944786 acc= 0.93333334
	  정답률 = 0.93333334
	  [1, 0, 0]
	  [1, 0, 0]
	  [1, 0, 0]
	  [1, 0, 0]
	  ....
	
