## tensorflow & keras
- Tensorflow : 구글의 오픈 소스 머신러닝 라이브러리
- Keras : Tensorflow 위에서 동작하는 라이브러리
⇒ Keras는 사용자 친화적이기때문에 Tensorflow를 사용하는 것보다는 사용이 훨씬 편하다.

## scikit-learn package 
다양한 머신러닝 모델(알고리즘) 제공
ex. 데이터 전처리(processing), 지도학습(supervised learning), 비지도학습(unsupervised learning), 모델 평가 및 선택(evaluation and selection)

## SVM(support vector machine)
분류 과제에서 사용할 수 있는 머신러닝 지도학습 모델
[https://github.com/bosl95/MachineLearning_Note/tree/master/0707](https://github.com/bosl95/MachineLearning_Note/tree/master/0707)

## Randome Forest
![enter image description here](https://i0.wp.com/hleecaster.com/wp-content/uploads/2020/01/rf01.png?w=1018)
 [의사결정나무(Decision tree)](https://dlsdn73.tistory.com/655) 머신러닝 알고리즘의 단점을 보완 ( 학습 데이터 오버피팅)
Training을 통해 구성해 놓은 다수의 나무들로부터 분류 결과를 취합해 결론을 얻음.
``scikit-learn`` 을 사용해 구현 가능.

## scikit-learn cross validation
고정된 test set을 가지고 모델의 성능을 확인하고 이 과정을 반복하여 만들어진 모델을 test set에만 잘 동작하는 모델이 된다. 즉, "test set에 과적합(overfitting)"된다는 것이다.
이를 해결하기 위한 것이 **Cross validation(교차 검증)** 이다.

![enter image description here](https://mblogthumb-phinf.pstatic.net/MjAxOTA3MjVfMTYw/MDAxNTY0MDYxOTQxODg2.2SJCkdADPvofL7LceWnSthfefB3UvnQ2_YoRp5F2vFog.4EZrViOF41rKfovPOJJMyv7W2HKTEvfDyg92pwIIIJ4g.PNG.ckdgus1433/image.png?type=w800)

- 모든 데이터 셋을 평가에 활용. 데이터의 편중을 막을 수 있다. (overfitting 방지)
- 모든 데이터 셋을 훈련에 활용할 수 있다. ⇒ 정확도 향상. 데이터 부족으로 인한 underfitting 방지

#### 01. cross validation 

	import pandas as pd  
	from sklearn import svm, metrics  
	from sklearn.model_selection import train_test_split  
	import sklearn.model_selection  
	  
	path = 'mushroom.csv'  
	mr = pd.read_csv(path, header=None)  
	label1 = [[0, 1], [1, 0]]  
	label = []  
	data = []  
	  
	for idx, row in mr.iterrows():  
	    la = row[0]  
	    label.append(ord(la))  
	    chs = []  
	    for ch in row[1:]:  
	        chs.append(ord(ch))  
	    data.append(chs)  
	  
	train_data, test_data, train_label, test_label = train_test_split(data, label)   
	  
	clf = RandomForestClassifier()
	clf.fit(train_data, train_label)  
	pre = clf.predict(test_data)  
	  
	score = metrics.accuracy_score(test_label, pre)  
	print(score)  
  
	score2 = sklearn.model_selection.cross_val_score(clf, data, label, cv=5)  
	# 교차 검증 (대상, 데이터, 정답, 그룹수)  
	print('cross : ', score2)  

	# SVC로 fit을 진행한 결과
	# 0.9935992122107337  
	# cross :  [0.82953846 0.99876923 0.91876923 1.         0.50184729]
	
	# RandomForest로 fit을 진행한 결과
	# 1.0  
	# cross :  [0.84246154 1.         1.         1.         0.64963054]

## 02. GridSearch
심층신경망의 대표적인  hyper parameter에는 학습률, 히든 레이어의 크기, 히든 레이어의 개수 등이 있다.
최적의 성능을 내기 위한 hyper parameter 탐색 방법이 있다.
1) Manual search : 사용자가 꼽은 조합내에서 최적의 조합을 찾는 것.
2) Grid search : hyper parameter에 적용할 값들을 미리 정해두고 모든 조합을 시행하여 최적의 조합을 찾는 방법.
3) Random search : hyper parameter의 최소-최대값을 정해두고 범위 내 무작위 값을 반복적으로 추출하여 최적의 조합을 찾는 방법.
Grid search는 사용자가 꼽은 선택지 중에서만 고르지만 Random search는 훨씬 다양한 조합들을 시험하여 예상치 못한 결과들을 얻을 수 있다.

	![enter image description here](https://www.kdnuggets.com/wp-content/uploads/hyper-parameter-search.jpg)

Grid search의 단점 ⇒ Hyper parameter 후보군 갯수만큼 비례하여 시간이 늘어나기 때문에 최적의 조합을 찾을 때까지 시간이 매우 오래 걸린다.

	train_label, train_data = load_csv('train.csv')  
	test_label, test_data = load_csv('t10k.csv')  
	  
	# hyper parmeter 후보군
	params = [{"C": [1, 10, 100, 1000], "kernel": ["linear"]},  
	  {"C": [1, 10, 100, 1000], "kernel": ["rbf"], "gamma": [0.001, 0.0001]}]  
	clf = GridSearchCV(svm.SVC(), params, n_jobs=-1)  
	clf.fit(train_data, train_label)  
	pre = clf.predict(test_data)  
	print(pre)  
	score = metrics.accuracy_score(test_label, pre)  
	print(score)

	'''
	출력
	[7 2 1 0 4 1 4 9 2 9 0 6 9 0 1 5 9 7 3 4 9 6 6 5 4 0 7 4 0 1 3 1 3 4 7 2 7
	1 2 1 1 7 4 2 5 5 1 2 4 4 6 3 5 5 2 0 4 1 9 5 7 2 9 3 7 4 2 4 3 0 7 0 2 7
	1 7 3 2 9 7 7 6 2 7 8 4 7 5 6 1 3 6 9 3 1 4 1 7 6 9 6 0 5 4 3 9 2 1 9 4 8
	7 3 9 7 9 4 4 9 2 5 4 7 6 4 9 0 5 8 5 6 6 5 2 8 1 0 1 6 4 6 7 3 1 7 1 8 2
	0 4 9 3 5 5 1 5 6 0 3 4 4 6 5 4 6 5 4 5 1 4 4 7 2 3 2 7 1 8 1 8 1 8 5 0 8
	4 2 5 0 1 1 1 0 3 0 3 1 6 4 2 3 6 1 1 1 3 9 5 2 9 4 5 9 3 9 0 3 5 7 5 7 2
	2 7 1 2 8 4 1 7 5 3 8 7 7 7 2 2 4 1 5 5 8 4 2 5 0 6 4 2 4 1 9 5 7 7 2 8 2
	6 8 1 7 7 9 1 8 1 5 0 3 0 1 9 9 9 1 8 2 1 2 9 7 5 9 2 6 4 1 5 3 2 9 2 0 4
	0 0 2 8 5 2 1 2 4 0 2 9 4 1 3 0 0 5 1 9 6 5 0 5 7 7 9 3 5 9 2 0 7 1 1 2 1
	5 3 2 9 7 0 6 5 4 1 3 5 1 0 5 1 3 1 5 5 6 1 8 5 1 9 9 4 6 7 2 5 0 6 5 6 3
	7 2 0 8 8 5 9 1 1 4 0 3 3 7 6 1 6 2 1 9 2 8 6 1 9 5 2 5 4 4 2 8 3 9 2 4 5
	0 3 1 7 7 3 7 9 7 1 9 2 1 4 2 9 2 0 4 9 1 9 8 1 8 4 8 9 7 8 3 7 6 0 0 3 5
	2 0 6 4 8 5 5 3 2 3 9 1 2 5 8 0 9 6 6 6 3 8 8 2 2 5 8 9 6 1 8 4 1 2 5 5 1
	9 7 7 4 0 4 9 7 1 6 5 2 3 7 0 9 4 0 6 3]
	0.872255489021956
	'''