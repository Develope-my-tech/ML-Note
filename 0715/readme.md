## 01. Text Processing
- Word Tokenization ( 단어 토큰화 )
	
	입력:  **Time is an illusion. Lunchtime double so!**
	출력 : "Time", "is", "an", "illustion", "Lunchtime", "double", "so"




## 02. Text Processing API

- one-Hot Encoding

	{'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}  
	
		one_hot_encoding("자연어",word2index)
		# [0, 0, 1, 0, 0, 0]  
		
한계 : one-hot encoding을 사용하면 단어 수가 늘어날 수록 기하급수적으로 늘어나기 때문에 비효율적이다.

    Ex) 강아지 = [ 0 0 0 0 1 0 0 0 0 0 0 0 ... 중략 ... 0] # 이 때 1 뒤의 0의 수는 9995개.
	이러한 벡터 표현은 공간적 낭비를 불러일으킵니다

이 문제를 해결하기 위한 word embedding 방법

one-hot encoding(=sparse representation)의 단점은 단어 간의 관계가 드러나지 않는 다는 점이다.
![enter image description here](https://files.slack.com/files-pri/T25783BPY-F6NNEPE01/one-hot.png?pub_secret=1e9eec95ff)
> **임베딩 모델(word embedding model)** : 단어의 의미를 최대한 담는 벡터를 만들려는 알고리즘

Dense representation : 각각의 속성을 우리가 정한 개수의 차원으로 대상을 대응시켜서 표현
![enter image description here](https://files.slack.com/files-pri/T25783BPY-F6P915890/dense.png?pub_secret=3f6e3ccd28)
위 그림에서 ‘강아지’란 단어는 [0.16, -0.50, 0.20. -0.11, 0.15]라는 5차원 벡터로 표현된다. 이 때 각각의 차원이 어떤 의미를 갖는지는 알 수 없다. 여러 속성이 버무러져서 표현되었기 때문이다. 다만 ‘강아지’를 표현하는 벡터가 ‘멍멍이’를 표현하는 벡터와 얼마나 비슷한지, 또는 ‘의자’를 표현하는 벡터와는 얼마나 다른지는 벡터 간의 거리를 통해 알 수 있다.

[Word Embedding 참고 자료](https://dreamgonfly.github.io/machine/learning,/natural/language/processing/2017/08/16/word2vec_explained.html)

- **Embedding()** : Embedding()은 단어를 밀집 벡터로 만드는 역할을 합니다. 인공 신경망 용어로는 임베딩 층(embedding layer)을 만드는 역할
	![enter image description here](https://user-images.githubusercontent.com/34594339/87533399-84659080-c6cf-11ea-8900-8780ce9d5d1f.png)


<STEP 4>

![enter image description here](https://user-images.githubusercontent.com/34594339/87533917-503e9f80-c6d0-11ea-87f6-c86ff417cc55.png)

<STEP 5>

![enter image description here](https://user-images.githubusercontent.com/34594339/87534231-b1ff0980-c6d0-11ea-9436-41d6e98a3ba1.png)

<STEP 6>

![enter image description here](https://user-images.githubusercontent.com/34594339/87534090-8845e280-c6d0-11ea-9f32-0218b8f66f31.png)

## 03. Pre-Trained Glove Embedding
<STEP 3>
glove.6B.100d.txt : 사전 훈련된 GloVe  파일.

    ['the', '-0.038194', '-0.24487', '0.72812', ... 중략... '0.8278', '0.27062']  # word_vector
    the 	# word
    [',', '-0.10767', '0.11053', '0.59812', ... 중략 ... '0.45293', '0.082577']  # word_vector
    , 		# word

<STEP 8>

![enter image description here](https://user-images.githubusercontent.com/34594339/87538653-829fcb00-c6d7-11ea-9454-30ca137ad448.png)

<STEP 9>

![enter image description here](https://user-images.githubusercontent.com/34594339/87538859-e629f880-c6d7-11ea-874d-297e45f31240.png)

*윗 방법으로 실행한 Dense보다 Embedding이 정확도가 떨어진다.*


## 04. 연습문제
1) [뉴스 분류하기](https://wikidocs.net/22933)
Dense와 Embedding 두가지 버전 확인해보기.
2) 가구 분류하기 (kaggle: furniture detector)
[https://www.kaggle.com/akkithetechie/furniture-detector](https://www.kaggle.com/akkithetechie/furniture-detector)