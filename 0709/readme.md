## 01. Practice
- heart.csv
	age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
	⇒ 심장병과 연관이 있는 요소들.
	- label : target

- data processing

	  #값의 범위를 0~1로 만들어줌  
	  csv['age']=csv['age']/100  
	  csv['trestbps']=csv['trestbps']/200  
	  csv['chol']=csv['chol']/500  
	  csv['thalach']=csv['thalach']/100

- training

	  clf = svm.SVC()
	  clf.fit(train_data, train_label)
	  pre = clf.predict(test_data)  
	  score = metrics.accuracy_score(test_label, pre)  
	  print(pre)  
	  print('score:', score)
	
- pickle 모듈을 이용해 학습 모델 저장, 불러오기

	  with open('heart_res.pkl', 'wb') as f:  
	    pickle.dump(clf, f)#오픈한 파일에 저장  
	  
	  with open('heart_res.pkl', 'rb') as f:  
	      clf = pickle.load(f)
	