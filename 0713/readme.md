참고
-   퍼셉트론(perceptron) 인공신경망의 기초개념
[https://blog.naver.com/samsjang/220948258166](https://blog.naver.com/samsjang/220948258166)
-   아달라인과 경사하강법(gradient descent)
[https://blog.naver.com/samsjang/220959562205](https://blog.naver.com/samsjang/220959562205)
-   딥러닝의 기초 - 다층 퍼셉트론(Multi-Layer Perceptron; MLP)
[https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=221030487369&parentCategoryNo=&categoryNo=87&viewDate=&isShowPopularPosts=true&from=search](https://blog.naver.com/PostView.nhn?blogId=samsjang&logNo=221030487369&parentCategoryNo=&categoryNo=87&viewDate=&isShowPopularPosts=true&from=search)
-   케라스(Keras) 기본 - 모델 학습, 히스토리 기능, 모델(신경망) 생성, 시각화 등
[https://m.blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221506748164&proxyReferer=https:%2F%2Fwww.google.co.kr%2F](https://m.blog.naver.com/PostView.nhn?blogId=qbxlvnf11&logNo=221506748164&proxyReferer=https:%2F%2Fwww.google.co.kr%2F)
  
  
 ## MLP(Multi-Layer Perceptron)
 [https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/](https://tykimos.github.io/2017/01/27/MLP_Layer_Talk/)
 ![뉴런](http://tykimos.github.io/warehouse/2017-1-27_MLP_Layer_Talk_neuron.png)
 -   x0, x1, x2 : 입력되는 뉴런의 축삭돌기로부터 전달되는 신호의 양
-   w0, w1, w2 : 시냅스의 강도, 즉 입력되는 뉴런의 영향력을 나타냅니다.
-   w0_x0 + w1_x1 + w2*x2 : 입력되는 신호의 양과 해당 신호의 시냅스 강도가 곱해진 값의 합계
-   f : 최종 합계가 다른 뉴런에게 전달되는 신호의 양을 결정짓는 규칙, 이를 활성화 함수라고 부릅니다.

	![enter image description here](http://tykimos.github.io/warehouse/2017-1-27_MLP_Layer_Talk_lego_2.png)
	![enter image description here](http://tykimos.github.io/warehouse/2017-1-27_MLP_Layer_Talk_lego_3.png)

## Dense Layer  
입출력을 모두 연결해주는 Layer.
예를 들어 입력 뉴런이 4개, 출력 뉴런이 8개 있다면 총 연결선은 32개입니다. 연결선이 32개이므로 가중치도 32개. 
  
- ``Dense(8, input_dim=4, init='uniform', activation='relu'))``  
	- 첫번째 인자 : 출력 뉴런의 수  
	- input_dim : 입력 뉴런의 수  
	- init : 가중치 초기화 방법 
		- 'uniform' : 균일 분포
		-  'normal' : 가우시안 분포
	- activation : 활성화 함수  
	   - ‘linear’ : 디폴트 값, 입력뉴런과 가중치로 계산된 결과값이 그대로 출력으로 나옵니다.  
	   - ‘relu’ : rectifier 함수, 은닉층에 주로 쓰입니다. ( 음수 데이터를 0으로 변환)  
	   - ‘sigmoid’ : 시그모이드 함수, 이진 분류 문제에서 출력층에 주로 쓰입니다. ( 0 ~ 1 사이의 수 반환 )  
	   - ‘softmax’ : 소프트맥스 함수, 다중 클래스 분류 문제에서 출력층에 주로 쓰입니다. (예측되는 결과값의 확률을 반환.)  
  
- ``compile()`` : 모델 학습 과정 설정. 손실 함수 및 최적화 방법을 정의. 모델을 기계가 이해할 수 있도록 컴파일 합니다.  
  
	- **optimizer** : 훈련 과정을 설정하는 옵티마이저를 설정합니다. 'adam'이나 'sgd'와 같이 문자열로 지정할 수도 있습니다.    
	- **loss** : 훈련 과정에서 사용할 손실 함수(loss function)를 설정합니다.    
	- **metrics** : 훈련을 모니터링하기 위한 지표를 선택합니다.  
     
- ``fit()`` :  모델이 오차로부터 매개 변수를 업데이트 시키는 과정을 학습, 훈련, 또는 적합(fitting)이라고 하기도 하는데, 모델이 데이터에 적합해가는 과정  
     
	> ##### 가중치가 높을수록 해당 입력 뉴런이 출력 뉴런에 미치는 영향이 크고, 낮을수록 미치는 영향이 적다.  

## 1. 케라스 개념을 위한 예제 1  
1) 학습 모델 생성

	   model = models.Sequential()

2) Dense Layer 생성

	입력층인 경우에만 input_shape 혹은 input_dim을 정의
	kernel_initializer: 가중치 초기화  
	bias_initializer: 바이어스 초기화

	   model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal',input_shape=(2,)))  
	   # model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal',input_dim=2)) 와 같음.

	입력층이 아닌 경우에는 이전 층의 출력 뉴런 수를 알 수 있기 때문에 input_dim을 정의하지 않아도 된다.

	   model.add(layers.Dense(4,activation='relu',kernel_initializer='random_normal',bias_initializer='random_normal'))

	  0 혹은 1을 나타내는 출력 뉴런 하나만 필요하므로 출력 뉴런이 1개이면서 입력 뉴런과 가중치를 계산한 값이 0에서 1사이를 표현할 수 있는 함수인 ``sigmoid 함수``를 사용한다. 

	   model.add(layers.Dense(1,activation='sigmoid'))

3) 모델 학습과정 설정

	    model.compile(optimizer='rmsprop',  
	    loss = 'binary_crossentropy', #0,1  
	    metrics=['accuracy'])

	- loss : 현재 가중치 세트를 평가하는 데 사용한 손실 함수. 이진 클래스 문제이므로 'binary_crossentropy'로 지정.
	- optimizer : 최적의 가중치를 사용하는데 최적화 알고리즘. rmsprop은 순환 신경망(RNN)의 옵티마이저로 많이 사용.
		- 'adam' : 효율적인 경사 하강법 알고리즘 중 하나.
	- metrics : 평가 척도. 분류 문제에서는 일반적으로 accuracy로 지정.

4) 모델 학습

	   history = model.fit(train_data, train_label,  
	     epochs = 20,  
	     batch_size=4,  
	     validation_data = (test_data,test_label))

	- epoch : 전체 훈련 데이터 셋에 대한 학습 반복 횟수 지정.
	- batch size : 가중치를 업데이트할 배치 크기.
	- validation data : epoch가 끝날 때마다 전달된 데이터의 손실과 측정 지표를 출력.

5) history(이력) 정보 

		his = history.history  
		k = his.keys()  
		print(k)
		dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

6) 결과
![image](https://user-images.githubusercontent.com/34594339/88989047-9ffaa900-d315-11ea-820d-fa146210d620.png)

## 2. 0 ~ 100까지의 점수 중 합격/불합격을 판단하는 예제  
합격/불합격 분류  
train_data = np.random으로 0에서부터 100사이의 값 100개...만들기  
train_label = 0,1,1,0….. =>합격 1, 불합격0
![image](https://user-images.githubusercontent.com/34594339/88989990-d6392800-d317-11ea-8ac6-eed6e0275358.png)

## 3. 케라스의 샘플 데이터 사용 (영화평 긍정, 부정). 사람들이 글을 올릴 때 많이 사용하는 단어 100개를 뽑아 사전을 구축.  
  
   **머신 러닝에서 텍스트 분류. 그 중에서도 특히, 감성 분류를 연습하기 위해 자주 사용하는 데이터가 있습니다. 바로 영화 사이트 ```IMDB의 리뷰 데이터```입니다. 이 데이터는 리뷰에 대한 텍스트와 해당 리뷰가 긍정인 경우 1을 부정인 경우 0으로 표시한 레이블로 구성된 데이터입니다**  

- 영화 리뷰 데이터 가져오기

	  (train_data, train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000)

	- load_data : 영화 리뷰 데이터를 가져오는 함수
	- num_words : 등장 빈도 순위로 몇번째 해당하는 단어까지 사용할 것인지 의미. 여기선 단어 집합의 크기를 10000으로 설정.

	      print(train_data[0]) # 0번째 리뷰 확인  
	      # [1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66 ... ] ( 리뷰에서 자주 나오는 단어들의 인덱스 )

- 레이블을 벡터로 변환
신경망에 숫자 리스트를 주입할 수 없으므로 가능한 형태로 변환한다.
여기서 단어 인덱스 범위가 10000 이므로 0 - 10000 사이의 단어들이 있는지 없는 지 판단할 수 있다.
예를 들어, sequence 중에 4번 18번이 있다면 그 인덱스는 1.0이 된다.
	
		# 훈련 데이터를 벡터로 변환  
		x_train = vectorize_sequences(train_data)  
		print(x_train[0])   # [0. 1. 1. ... 0. 0. 0.]  
		# 테스트 데이터를 벡터로 변환  
		x_test = vectorize_sequences(test_data)

	1) 정수 텐서로 변환
	2) one-hot encoding ( =categorical encoding, 범주형 인코딩)

- 학습 과정
앞에서 했던 과정들과 똑같다.

- test
test_data(리뷰 리스트)의 리뷰 10개를 추출(0번째 방부터) 문장으로 변환  
이를 predict하여 긍정 문장인지 부정 문장인지 출력

	  word_index = imdb.get_word_index()
	  reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
	  # [ {단어:정수}, .... ] 로 이루어진 dictionary를 불러와 {단어 : 정수} 를 {정수 : 단어} 로 바꾼다.

	0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺀다.
	0, 1, 2를 가진 인덱스가 나올 경우 '?'로 대체, 나머지는 인덱스에 맞는 단어를 출력한다.

		# 0, 1, 2는 '패딩', '문서 시작', '사전에 없음'을 위한 인덱스이므로 3을 뺍니다

	![](https://lh5.googleusercontent.com/Y8mOx47-0Dg7iBo9PZgNQ-ZGsz57fJFUFcXA6C1QodL3EKjuVlFcAg9PT47uDosJ7s486njRKtJJJEys5n28-LwCBHTKPpBoMxcR1afy2CovL91aeju_Id8JZW2XKPZB4g)

- loss / accuracy
	![image](https://user-images.githubusercontent.com/34594339/88991572-fa970380-d31b-11ea-9ca3-75e9560bba5a.png)
	![image](https://user-images.githubusercontent.com/34594339/88991579-fff44e00-d31b-11ea-87cb-e6b5c7d39a82.png)


	![image](https://user-images.githubusercontent.com/34594339/88992042-41392d80-d31d-11ea-9e1a-5db1e1438a4d.png)
  
  
5. bmi.csv 파일을 이용한 비만도 측정.  
- Dropout : 학습데이터에 대한 "과적합(overfitting)"을 방지하기 위한 장치
![image](http://cs231n.github.io/assets/nn2/dropout.jpeg)

	  model.add(Dropout(0.1))

- 결과
![image](https://user-images.githubusercontent.com/34594339/88992488-84e06700-d31e-11ea-8705-fd61af2a188f.png)

6. 손글씨 숫자 인식
[https://potensj.tistory.com/13](https://potensj.tistory.com/13)

		network.compile(optimizer = 'rmsprop',  
		  loss = 'categorical_crossentropy',  
		  metrics = ['accuracy'])

	- categorical_crossentropy : 다중 분류 손실 함수

- 이미지 데이터 준비
mnist 이미지 데이터들은 흑백이지만, 진하기에 따라 0 ~ 255까지의 숫자로 지정되어 있으므로 0 ~ 1 사이의 값으로 Nomalize 해준다. ( float32로 변환하는 것도 이때문이다.)

	  train_images = train_images.reshape((60000,28*28))  
	  train_images = train_images.astype('float32')/255  
	  
	  test_images = test_images.reshape((10000,28*28))  
	  test_images = test_images.astype('float32')/255

- Preprocess class labels
label data 0 ~ 9 를 one-hot encoding으로 표현
5 ⇒ [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  

	  train_labels = to_categorical(train_labels) # one hot encoding  
	  test_labels = to_categorical(test_labels)

- 출력

	  test_loss, test_acc = network.evaluate(test_images, test_labels)  
	  print('test_acc : ', test_acc)
	  # test_acc :  0.9781000018119812