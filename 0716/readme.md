
## RNN (Recurrent Neural Network, 순환 신경망)
현재 입력 데이터와 과거 데이터를 고려하여 순차 데이터를 처리.

- RNN 구조 예시  
	![enter image description here](https://files.slack.com/files-pri/T25783BPY-F6XAEQH7T/rnn-examples.png?pub_secret=0eb724d01b)  
	  
	1. 고정 크기 입력, 고정 크기 출력. --> 단순한 형태의 RNN(**vanilla RNN**)  
	2. 고정 크기 입력, 시퀀스 출력.   ex) 이미지를 입력해 이미지에 대한 설명을 문장으로 출력하는 이미지 캡션  
	  
	![enter image description here](https://files.slack.com/files-pri/T25783BPY-F6XPSQ4NP/show-and-tell.png?pub_secret=1e78ced420)  
	  
	3. 시퀀스 입력, 고정 크기 출력.   ex) 문장을 입력해서 긍부정 정도를 출력하는 감성 분석기  
	4. 시퀀스 입력, 시퀀스 출력.    ex) 영어를 한국어로 번역하는 자동 번역기  
	5. 동기화된 시퀀스 입력, 시퀀스 출력.    ex) 문장에서 다음에 나올 단어를 예측하는 언어 모델

- RNN의 기본 구조
	![enter image description here](http://i.imgur.com/Q8zv6TQ.png)
	![](http://i.imgur.com/vrD0VO1.png)

	- 빨간 박스 : input x
	- 파란 박스 : output y
	- 녹색 박스 : hidden state
	hidden state $`h_{t}`$는 직전 시점의 state $h_{t−1}$를 받아 갱신된다.
	
		> $h_{t} = tanh(W_{hh}h_{t-1}+W_{wh}x_{t}+b_{h})$
	
		히든 state의 활성함수(activation function)은 비선형 함수인 하이퍼볼릭탄젠트(tanh)이다.
	

## 01. RNN_imdb
[https://aciddust.github.io/blog/post/Keras-%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D-%EC%9D%B4%ED%95%B4/](https://aciddust.github.io/blog/post/Keras-%EC%88%9C%ED%99%98%EC%8B%A0%EA%B2%BD%EB%A7%9D-%EC%9D%B4%ED%95%B4/)

- SimpleRNN : 가장 간단한 형태의 RNN Layer
	![image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2F4o8ga%2FbtqCNHoLYJi%2FhluVF6iCJrN8ANEGzIqRj0%2Fimg.png)
	
	SimpleRNN은 시퀀스 batch 처리를 한다.
	
1) Input 시퀀스에 대한 마지막 출력만 반환 : (batch_size, output_features) 크기의 2D 텐서.

		model = Sequential()  
		model.add(Embedding(10000, 32))
		# Embedding(단어 집합의 크기, 임베딩 벡터의 출력 차원(결과로 나오는 임베딩 벡터 크기), input_length(입력 시퀀스의 길이))
		model.add(SimpleRNN(32))
		model.summary()

 - Layer 정보

		Model: "sequential_1"
		_________________________________________________________________
		Layer (type)                 Output Shape              Param #   
		=================================================================
		embedding_1 (Embedding)      (None, None, 32)          320000    
		_________________________________________________________________
		simple_rnn_1 (SimpleRNN)     (None, 32)                2080      
		=================================================================
		Total params: 322,080
		Trainable params: 322,080

2) 네트워크의 표현력을 증가시키기 위해 여러개의 순환층을 차례대로 쌓는 것이 유용할 수 있음. 이런 설정에서는 중간 층들이 전체 출력 시퀀스를 반환하도록 설정해야한다.

		model = Sequential()  
		model.add(Embedding(10000, 32))  
		model.add(SimpleRNN(32, return_sequences=True))  
		model.add(SimpleRNN(32, return_sequences=True))  
		model.add(SimpleRNN(32, return_sequences=True))  
		model.add(SimpleRNN(32))  # 맨 위 층만 마지막 출력을 반환.  
		model.summary()

- 출력

		Model: "sequential_2"
		_________________________________________________________________
		Layer (type)                 Output Shape              Param #   
		=================================================================
		embedding_2 (Embedding)      (None, None, 32)          320000    
		_________________________________________________________________
		simple_rnn_2 (SimpleRNN)     (None, None, 32)          2080      
		_________________________________________________________________
		simple_rnn_3 (SimpleRNN)     (None, None, 32)          2080      
		_________________________________________________________________
		simple_rnn_4 (SimpleRNN)     (None, None, 32)          2080      
		_________________________________________________________________
		simple_rnn_5 (SimpleRNN)     (None, 32)                2080      
		=================================================================
		Total params: 328,320
		Trainable params: 328,320
		Non-trainable params: 0
		_________________________________________________________________

3) 데이터 전처리
- 출력

		25000 훈련 시퀀스
		25000 테스트 시퀀스
		시퀀스 패딩 (samples x time)
		input_train 크기: (25000, 500)
		input_test 크기: (25000, 500)

4) Embedding 층과 SimpleRNN 층을 사용해 훈련.
5) 훈련 검증 및 손실 확인
![image](https://user-images.githubusercontent.com/34594339/89147494-c3318c80-d591-11ea-9dc9-3361780ff371.png)

처음 500개의 단어만 입력에 사용했기 때문에 정확도가 낮다.
``SimpleRNN`` 이 텍스트와 같이 긴 시퀀스를 처리하는데 적합하지 않다.
짧은 시퀀스에 대해서 효과가 있다.
time-step이 길어질수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생.

![enter image description here](https://wikidocs.net/images/page/22888/lstm_image1_ver2.PNG)  
  
바닐라 RNN은 출력 결과가 이전의 계산 결과에 의존. **비교적 짧은 시퀀스에 대한 효과를 보인다는 단점**이 있음  
$x_{1}$에서 $x_{t}$로 갈수록 앞의 정보가 손실 되어 가는 것을 볼 수 있다.  
--> **장기 의존성 문제(the problem of Long-Term Dependencies)**  

## 02. LSTM_imdb
- LSTM 구조
![image](http://i.imgur.com/jKodJ1u.png)

	기존 RNN의 단점을 보완.
	은닉층의 메모리 셀에 [입력, 망각, 출력] 게이트를 추가하여 불필요한 기억 제거 후 기억해야할 것들을 정한다.

	> 케라스는 좋은 기본값을 가지고 있어서 직접 매개변수를 튜닝하는 데 시간을 쓰지 않고도 거의 항상 어느 정도 작동하는 모델을 얻을 수 있다.
	  
	- 바닐라 RNN의 구조  
		![enter image description here](https://wikidocs.net/images/page/22888/vanilla_rnn_ver2.PNG)  
	  
	- LSTM의 구조  
	   - 이하 식에서 σ는 시그모이드 함수를 의미합니다.  
	   - 이하 식에서 tanh는 하이퍼볼릭탄젠트 함수를 의미합니다.  
	  
		![enter image description here](https://wikidocs.net/images/page/22888/vaniila_rnn_and_different_lstm_ver2.PNG)


1) step 3 까지는 이전 과정과 동일.
2) Train

		from keras.layers import Dense, LSTM  
		  
		model = Sequential()  
		model.add(Embedding(max_features, 32))  
		#model.add(SimpleRNN(32))  
		model.add(LSTM(32, return_sequences = False))  
		# 가중치 손실을 막기 위해 운반 정보도 파라메터로 전달  
		model.add(Dense(1, activation='sigmoid'))  
		  
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])  
		history = model.fit(input_train, y_train,  
		  epochs=10,  
		  batch_size=128,  
		  validation_split=0.2)

3) 결과 출력
	![image](https://user-images.githubusercontent.com/34594339/89148425-bf533980-d594-11ea-9858-f8c2cb222fcb.png)

⇒ 검증 정확도가 많이 올라감. 과거의 데이터를 기억하고 있기 때문에 앞의 선택에도 영향을 줌. 그렇기 때문에 검증 시 정확도가 많이 올라가는 것으로 보인다.

## 03.BayesianFilter

anaconda 환경에서는 그냥 konlpy 모듈을 설치만 해서는 실행이 되지 않는다.
[해결 방법](https://m.blog.naver.com/PostView.nhn?blogId=hs_715&logNo=221548450575&proxyReferer=https:%2F%2Fwww.google.com%2F)

- KoNLPy : 한국어 정보처리를 위한 파이썬 패키지

		from konlpy.tag import Okt  
		  
		twitter = Okt()  
		data = twitter.pos('나는 오늘 카페에서 코딩을 하고 있다.', norm=True, stem=True)  
		for d in data:  
		    print(d)

		# 출력
		('나', 'Noun')
		('는', 'Josa')
		('오늘', 'Noun')
		('카페', 'Noun')
		('에서', 'Josa')
		('코딩', 'Noun')
		('을', 'Josa')
		('하다', 'Verb')
		('있다', 'Adjective')
		('.', 'Punctuation')

- 베이즈 정리 : 조건부 확률 P(A|B)는 사건 B가 발생한 경우 A의 확률
	
	> ### $P(A|B)=\frac{P(A\bigcap B)}{P(B)}$

	- 나이브 베이즈(Naive Bayes)
	
		> ### $P(A|B)=  \frac{P(B|A)\cdot P(A)}{P(B)}$

	참고자료
	[https://gomguard.tistory.com/69](https://gomguard.tistory.com/69)
	[https://wikidocs.net/22892](https://wikidocs.net/22892)

- 변수 확인

		print(bf.words)
		print(bf.word_dict)
		print(bf.category_dict)

		# 출력
		{'진행', '인기', '쿠폰', '제품', '소식', '기간', '세', '등록', '함께', '따뜻하다', '자다', '봄', '확인', '일정', '보고', '회의', '부탁드리다', '무료', '계약', '되어다', '파격', '배송', '오늘', '할인', '찾아오다', '프로젝트', '신제품', '없다', '한정', '현', '데', '일', '계', '30%', '백화점', '선물', '상황'}
		{'광고': {'파격': 1, '세': 3, '일': 3, '오늘': 1, '30%': 1, '할인': 1, '쿠폰': 1, '선물': 1, '무료': 1, '배송': 1, '현': 1, '데': 1, '계': 1, '백화점': 1, '봄': 1, '함께': 1, '찾아오다': 1, '따뜻하다': 1, '신제품': 1, '소식': 1, '인기': 1, '제품': 1, '기간': 1, '한정': 1}, '중요': {'오늘': 2, '일정': 3, '확인': 1, '프로젝트': 1, '진행': 1, '상황': 1, '보고': 1, '계약': 1, '자다': 1, '부탁드리다': 1, '회의': 1, '등록': 1, '되어다': 1, '없다': 1}}
		{'광고': 5, '중요': 5}

- 텍스트 학습

	  def fit(self, text, category):  
		  word_list = self.split(text)  
		  for word in word_list:  
		        self.inc_word(word, category)  	# 단어를 카테고리에 추가, 계산하는 함수
		  self.inc_category(category)					# 카테고리 빈도 계산하는 함수

- 텍스트 예측

	  def predict(self, text):  
		  best_category = None  
		  max_score = -sys.maxsize  
		  words = self.split(text)  					# 형태소 분석 (어미/조사/구두점 제외)
		  score_list = []  
		  for category in self.category_dict.keys():  
		      score = self.score(words, category)  			# 단어 리스트에 점수 매기기
		      score_list.append((category, score))  
		      if score > max_score:  
		          max_score = score  
		          best_category = category  
		  return best_category, score_list


		  def score(self, words, category):  
			    score = math.log(self.category_prob(category))  
			    for word in words:  
			        score += math.log(self.word_prob(word, category))  
			    return score
			# 확률을 곱할 때 값이 너무 작으면 다운 플로우가 발생할 수 있어 math.log 사용

- 결과 :  단어의 빈도수를 통해 카테고리를 구분.

		결과 = 광고
		[('광고', -19.485641988358296), ('중요', -20.63806741338132)]
