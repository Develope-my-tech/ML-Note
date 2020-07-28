
###  01. SVM (Support Vector Machine)  
- SVM : 결정 경계 (Decision Boundary, 분류)를 위한 기준 선을 정의하는 모델.  
- SVM에서는 ``커널트릭(kernel trick)``이라는 기법을 사용.  
 > 커널 트릭이란?  
 > 원본 데이터 차원에서는 선형으로 분리할 수 없었던 데이터를 커널 함수를 이용하여 고차원 특징 공간으로 이동, 선형으로 분리 가능한 형태로 바꿈. > SVM 커널 종류 : 선형(linear), 다항식(polynomial), 방사 기저 함수(radial basis function), 시그모이드(sigmoid), 지수 카이 제곱, 히스토그램 교채 ( 디폴트는 linear라고 함.)  
- 2차원에서의 결정 경계  
  
![enter image description here](https://i0.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm01.png?w=1372)  
  
- 3차원에서의 결정 경계  
  
![enter image description here](https://i0.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm02.png?resize=1536,1278)  
  
- 결정 경계가 3차원 이상인, 고차원의 형태를 ``초평면(hyperplane)``이라고 부른다.   
  
### **이상치(Outlier)**  
  
![enter image description here](https://i1.wp.com/hleecaster.com/wp-content/uploads/2020/01/svm06.png?w=1280)  
  
⇒ 위(Hard Margine) / 아래(Soft Margine)  
- hard margin의 경우 서포트 벡터와 결정 경계 사이의 거리가 매우 좁다.  ⇒ **overfitting** 발생  
- soft margin의 경우 서포트 벡터와 결정 경계 사이가 멀어지면서 마진이 커진다.  ⇒ **underfitting** 발생  
  
  
## 02. SVM을 이용한 MNIST 문자 인식  
- **mnist**  
 - 바이너리 파일  
      train-labels-idx1-ubyte : 학습  
      t10k-labels-idx1-ubyte : 테스팅  
   - 이미지 파일  
      train-images-idx3-ubyte : 학습  
      t10k-images-idx3-ubyte : 테스팅  
​  
1) 학습용 csv파일 생성하기.  
​  
  - struct 모듈 : 바이너리 데이터 ⇒ int / char / float 등의 다양한 타입으로 변환.  
  - struct.unpack(format, 바이너리값) : 원하는 바이너리 수 만큼 읽어 들이기 + 정수 변환a  
     
         # 헤더 정보 읽어오기  
		 mag, lbl_count = struct.unpack('>II', lbl_f.read(8))  # 레이블파일에서 매직넘버와 개수를 읽음         
		 mag, img_count = struct.unpack('>II', img_f.read(8))  # 숫자 이미지파일에서 매직넘버와 개수를 읽음    
         row, col = struct.unpack('>II', img_f.read(8))  # 숫자 이미지파일에서 이미지 가로, 세로 길이 읽음  
         
	  - format  
	         - '>' : 빅 엔디안 ( 바이트 열의 순서로 빅 엔디안은 저장 곤간을 큰 주소에서 부터 사용하겠다는 것. 엔디안이 다른 프로그램끼리는 종류를 동일하게 처리해주어야함.)  
	         - 'I' : unsigned int(4)  
	         - 'B' : unsigned char(1)  
​  
   - 이미지 데이터 읽고 csv로 저장하기   
     
			for idx in range(lbl_count):    
				if idx > maxdata:  # 1000이 넘으면 break    
					break    
				label = struct.unpack("B", lbl_f.read(1))[0] # 정답 파일(레이블)에서 숫자 한개씩 읽음    
				bdata = img_f.read(px)  # 숫자 이미지 파일에서 이미지 한 개 크기만큼 읽어서 bdata에 담음.    
				sdata = list(map(lambda n: str(n), bdata))    
				# print(sdata)     
				csv_f.write(str(label) + ',')          
				csv_f.write(','.join(sdata) + '\r\n')  
​  
​  
   - 결과  
      - train.csv : 학습용 파일  
      - train-?-?.pgm : 처리된 바이너리 파일을 이미지 파일로 변환. (이미지를 단위별로 잘 읽어왔는지 확인용)  
  
2) TESTING  
     
	   with open(path+'train.csv', 'r') as f:    
          for line in f:    
              l = line.split(',')    
              if l[0] == '\n':    
                  continue    
	          train_label.append(int(l[0]))    
	          tmp = list(map(int, list(l[1:])))    
	          train_data.append(tmp)    
	             
         train_data, test_data, train_label, test_label = train_test_split(train_data, train_label)    
             
         clf = svm.SVC()    
         clf.fit(train_data, train_label)    
         pre = clf.predict(test_data)    
         score = metrics.accuracy_score(test_label, pre)    
         print(pre)    
         print('score:', score)  
	     ''' 출력         
	     [5 7 5 8 8 4 2 5 2 3 4 1 4 3 4 7 1 3 1 1 0 5 5 9 0 8 1 8 9 9 5 5 4 0 5 3 1 6 0 8 5 1 0 4 8 0 4 2 3 9 9 9 2 0 8 0 9 1 6 7 7 5 6 9 4 6 0 3 5 0 5 0 7 5 8 9 8 8 0 2 0 3 5 0 6 7 8 7 6 0 4 7 4 1 9 5 1 5 0 7 8 1 6 7 5 4 9 4 4 9 9 9 2 6 1 7 7 8 5 2 2 3 1 6 0 9 7 5 9 1 5 9 2 1 9 7 4 5 6 2 2 9 7 2 8 6 4 6 4 4 9 4 5 6 6 7 2 7 5 1 7 3 1 7 3 9 7 8 9 8 5 5 2 8 9 9 5 2 7 3 7 1 6 1 4 7 0 4 0 9 3 1 5 9 8 4 8 3 9 4 4 3 7 0 3 8 7 2 5 8 7 7 0 7 0 5 7 3 8 1 3 7 2 7 5 9 1 2 9 2 0 6 9 1 1 3 3 8 7 8 5 1 0 1 7 7 7 1 6 8 6] 
		score: 0.9203187250996016 '''
	​  
​  
## 3. BMI 측정  
- bmi.csv : height,weight,label 순으로 저장.  
- 긴 설명이 필요없는 간단한 학습   
     
- Code  

	  csv = pd.read_csv('bmi/bmi.csv')
	  csv_data = csv[['height', 'weight']]   
	  csv_label = csv['label']    
	  train_data, test_data, train_label, test_label = train_test_split(csv_data, csv_label)    
         
      clf = svm.SVC()    
      clf.fit(train_data, train_label)    
      pre = clf.predict(test_data)    
         
      score = metrics.accuracy_score(test_label, pre)    
      print(pre)    # ['thin' 'normal' 'normal' ... 'thin' 'thin' 'thin']  
	  print(score)      # 0.992  ​  
​  
## 04. Alpabet 인식  
1) Alpabet 이미지 생성  
알파벳 한줄로 입력한 이미지를 잘라서 알파벳 데이터 셋으로 구성.  

	    gray_img = cv2.imread('alph/alphabets.png', 0)    
		al = 'abcdefghijklmnopqrstuvwxyz'    
		h, w = gray_img.shape    
		al_w = w/26    
		ww = 0    
		imgs = []    
     
	    for i in range(0, 26):    
		    img = gray_img[0:h, int(i*al_w):int(ww+al_w)]    
		    ww += al_w    
		    imgs.append(img)  
​  
2) 알파벳 바이너리 파일 ⇒ csv 파일로 만들어서 저장  
   - ``ravel()`` : 다차원 배열을 1차원으로 평평하게 만들어주는 함수.  
        
         # alpabet 사진으로 csv파일 생성.  
		 idx = 0         
		 csv_f = open('alph/al.csv', 'w', encoding='utf-8')    
         for x in imgs:    
             fdata = x.ravel()    
             img_data = list(map(lambda n: str(n), fdata))    
             csv_f.write(al[idx] + ',')     #  label 추가  
			 csv_f.write(','.join(img_data) + '\r\n')  # image data 추가 
			 idx+=1​  
			 
   - al.csv   
  
		 a, 255, 255, ...  
		 b, 255, 255, ...  
		 c, 255, 255, ...  
​  
      ⇒ 이런 형태의 파일로 저장되어있는 상태  
3) 학습 데이터로 Train & Test  

    path = 'alph/'    
    train_label = []    
    train_data = []    
          
    with open(path+'al.csv', 'r') as f:    
        for line in f:    
            if len(line)==1:    
                continue    
          l = line.strip().split(',')    
            train_label.append(l[0])    
            train_data.append([int(a) for a in l[1:]])    
        
    train_data, test_data, train_label, test_label = train_test_split(train_data, train_label)    
          
    clf = svm.SVC()    
    clf.fit(train_data, train_label)    
    pre = clf.predict([train_data[0]])  # ex) 'a'가 잘 인식 되는가?  
	print(pre)     # 출력 : 'a'