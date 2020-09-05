

# [목차]
###  1. [Darknet이란 무엇인가? ](#darknet)
### 2. [YOLO란 무엇인가? ](#yolo)
### 3. [YOLOV4 실행해보기 ](#yolov4-실행해보기)
### 4. [커스텀 모델로 학습모델 만들기 ](#custom-dataset을-이용한-학습모델-구현)
### 4-1. [데이터 가공하기 ](#1-데이터-가공하기)
### 4-2.  [훈련을 위한 cfg 설정](#2-훈련시키기-위한-설정)
- [기본지식 ](#기본-지식)
- [Google의 COLAB ](#colab)
- [custom data train을 위한 파일 ](#custom-data-train을-위한-파일)
### 4-3. [학습 모델 구현 과정](#3-학습-모델-구현하기)
### 4-4. [세부 설정 조정](#4-세부-작업)

## darknet
c언어로 작성된 물체 인식 오픈 소스 신경망

## YOLO
darknet을 통해 학습된 신경망  

이전 탐지 시스템은 classifier나 localizer를 사용해 탐지를 수행합니다.

하지만 YOLO는 하나의 신경망을 전체 이미지에 적용합니다.

이 신경망은 이미지를 영역으로 분할하고 각 영역의 Bounding Box와 확률을 예측합니다.

이런 Bounding Box는 예측된 확률에 의해 가중치가 적용됩니다.
  
## yolov4 실행해보기
1) dark net의 weight ⇒ yolov4.weights 으로 변환하는 과정.    

	```
	# Convert darknet weights to tensorflow
	 ## yolov4  버전
	 python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

	 ## yolov4-tiny 버전
	 python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny
	```
2) object detection이 잘 되는 지 확인하기	  
	```
	# Run demo tensorflow
	 python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg

	 python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/kite.jpg --tiny
	```
3) 결과  
   ![img](https://github.com/kairess/tensorflow-yolov4-tflite/raw/master/result.png)  
  
   ![image](https://user-images.githubusercontent.com/34594339/89185473-3f998f00-d5d5-11ea-99f7-45c37f85e8f0.png)  
  
   ⇒ yolov4 weight (위) / yolo4-tiny (아래)  
   속도는 tiny가 훨씬 빠르다.  

### ⇒ YOLOV4를 이용해 커스텀 데이터 셋을 만들려고 하였으나, YOLOV3를 이용한 정확도가 훨씬 높아 YOLOV3-tiny를 사용하기로 함.
  <br>
  
# Custom Dataset을 이용한 학습모델 구현

## 1) 데이터 가공하기

[AI Hub 데이터셋](http://www.aihub.or.kr/aidata/136)을 이용하여 신호등이 있는 사진과 Bounding Box가 되어있는 xml파일을 받았으나, **보행등과 차량등이 분류가 되어있지 않다** 는 문제점이 발생하였다.  
- 해결방안  
   1) 우선적으로 데이터 셋에 신호등이 다 있는 것도 아니기 때문에 1차적으로 신호등을 찾아준다.  
   ⇒ 신호등 label을 갖는 사진을 분류한다는 의미  
   2) label을 확인함과 동시에 신호등 사진을 띄운다. 그 사진속에 있는 신호등이 보행등이라면 저장, 차량등이라면 넘어간다.  
   3) 사진을 저장하는 경우에 label 데이터는 가공이 필요하다.  
   현재 AI hub에서 제공되는 Bounding Box  좌표 ⇒ (좌상단 x, 좌상단 y, 우하단 x, 우하단 y좌표)  
   yolo에서 데이터 셋을 훈련시킬 때 label의 좌표 ⇒ (center x, center y, ratio w, ratio h) 좌표  
           
        
 > #### COCO 데이터 포맷은 bouding box 값이  **x, y, w, h**  값으로 구성되어있다.하지만 YOLO에서의 포맷은 클래스 번호와  **전체 영상 크기에 대한 center x, center y, w, h 비율 값**으로 구성된다.  
  
 -  출처: [https://eehoeskrap.tistory.com/367](https://eehoeskrap.tistory.com/367)     
 - 참고 : [<보행등 사진만 분류하기>](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py) 
 - [PILLOW와 cv2의 shape 차이] : https://note.nkmk.me/en/python-opencv-pillow-image-size/  
 - [https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig](https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig)       
내가 사용한 데이터셋은 COCO 데이터는 아니지만 이 글들을 보면서 참고해서 데이터 포맷을 맞춰줬다.  
  
	 추가)  OpenCV 좌표계 
	 
	 ![enter image description here](https://lh4.googleusercontent.com/ndFH6A225tFLWb7JwjyMmn539c4e1c1CmU7w4hQD6j-uO9K4diKfZ-FDr8LFuKa9oad9IaunhXRz0kD0JoRbeRV4gzUpS0ELyPKMIlpXs9FgvbJZiNGreGvWQAlMnYnRkqzo8Vlh)  


-  ### [데이터 라벨링 하기 ](https://github.com/bosl95/MachineLearning_Note/tree/master/contest_preparation/labeling  )
  
## 2) 훈련시키기 위한 설정

>  ## 기본 지식
 ![image](https://user-images.githubusercontent.com/34594339/89891485-1e0d5880-dc10-11ea-8b08-4c61505a6bf6.png)
 
> ##  Colab 
  - yolo를 노트북에서도 사용하기 위해서는 **GPU를 사용해야 한다.**   
  - 이를 위해서 Google에서 지원하는 Colab을 이용해 yolo를 구동시킬 수 있다.  
   - Colab을 세션을 12시간만 유지시켜주기 때문에 저장이 불가하다. ==> 구글 드라이브에 데이터를 저장해 놓고 마운트 해서 쓸 수 있다
   -  주피터 노트북이 명령 프롬트에서 입력한  것처럼  처리하는 명령어  
	   - `` !`` :  쉘이 끝나면 유지 되지 않음
	   - ``%`` : 쉘이 끝난 후에도 계속 유지
	   
![image](https://user-images.githubusercontent.com/34594339/89725910-db9d1d80-da4f-11ea-88bf-8ab79c47a555.png)  
  
> ## custom data train을 위한 파일  
###  1) ```obj.data``` : 학습을 위한 내용이 담긴 파일  
   - classes 개수  
   - train.txt와 valid.txt의 경로  
   - obj.names의 경로  
   - weight을 저장할 폴더의 경로  
###   2) ``obj.cfg``
   - 모델 구조 및 train과 관련된 설정이 들어있는 파일  
   - batch 및  subdivisions 사이즈(Cuda memory 관련), width 및 height 사이즈  
   - learning late, burn_in, max_batches,  policy, steps, scales 설정  
   - filter : (4+1+class수) * 3  
   - classes  
   - anchors 및 mask 설정  
### 3) ``weight``  : 미리 트레이닝 된 모델 또는 darknet53.conv.74 등의 가중치 파일  
### 4) ``obj.names`` : annotation에 포함되어있는 라벨링 이름 목록. 검출하고자 하는 목록  
###  5) ``images`` : 학습시킬 이미지들   
### 6) ``annotation`` : 학습시킬 이미지에 대한 주석들  
   - 각 이미지마다 주석들이 담긴 텍스트파일이 필요  
      - 0001.txt  
      - 0002.txt  
      ....  
### 7) ``train.txt`` : 학습시킬 이미지들의 경로들이 담긴 리스트  
### 8) ``valid.txt`` : 학습 시 validation 할 이미지들의 경로들이 담긴 리스트  
  
---

## 3) 학습 모델 구현하기  
 
1) cfg 설정  
   - custom_yolov4-tiny.cfg 파일 수정  
   ![image](https://user-images.githubusercontent.com/34594339/89791590-7b48d180-db5e-11ea-9c98-5e67e557fc33.png)  
  
   - anchor 계산  
      ![image](https://user-images.githubusercontent.com/34594339/89791801-b9de8c00-db5e-11ea-9e7a-b9e63bdbe049.png)  
  
      - 참고자료  
         - [cfg 설정을 위한 설명글](https://eehoeskrap.tistory.com/370)  
         - [YOLO v4 custom데이터 훈련하기](https://keyog.tistory.com/22)  
         - [커스텀 데이터 셋으로 Yolo 써 보기 1](https://jueun-park.github.io/2018-07-12/yolo-custom-dataset)  
         - [https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)  
           
      - 추가 참고자료  
         - [https://murra.tistory.com/115](https://murra.tistory.com/115) ⇒ 가장 도움을 많이 받은 자료.  
         - [https://codingzzangmimi.tistory.com/76](https://codingzzangmimi.tistory.com/76)  
         - [https://go-programming.tistory.com/160](https://go-programming.tistory.com/160)  
  
        
2) train 명령어  
      
      	!./darknet detector train custom/custom.data custom/custom_yolov4-tiny.cfg custom/yolov4-tiny.conv.29 -dont_show  
          
     - 원래 map과 loss에 대한 그래프가 나오는데 코랩의 리눅스 상에서는 볼 수 없는 듯하다. 에러가 나기 때문에 dont_show를 추가해 보지 않는 것으로 처리해준다.  
 - yolov4-tiny.conv.29 : pre-train된 weight 값을 넣어주었다. 첫 training에서 비워두고 사용해도 된다고 함.  
  
3) detect 명령어  
        
       !./darknet detector test custom/custom.data custom/custom_yolov4-tiny.cfg custom_yolov4-tiny_last.weights -thresh 0.25 -dont_show -ext_output < custom/train.txt > result.txt  
 - 이때 tarin.txt에 있는 이미지의 경로를 읽어오지 못한다는 에러가 발생했다.  
       
      !apt-get install dos2unix   
       !dos2unix custom/train.txt  # to linux format  
  train.txt 파일을 윈도우상에서 만들었기 때문에 dos2unix라는 모듈을 이용하여 txt파일을 리눅스상에서 읽을 수 있는 포맷으로 바꾸어주었다.  
       
   - 실행결과  
       
      ![image](https://user-images.githubusercontent.com/34594339/89888430-ea7bff80-dc0a-11ea-8cb0-6601663528bc.png)  
        
     weight를 학습하던 중에 colab 세션이 종료되어서 학습을 제대로 끝마치지 못한 상태였는데,  
     마지막에 train된 weight 파일로 detection을 한 결과 정확도는 많이 떨어지지만 신호등 객체를 탐지하는 것을 확인할 수 있었다.
     
 
## 4) 세부 작업

<details>
<summary>training에서 정확도 올리기</summary>
<br>

- width랑 height가 클수록 정확도는 올라감
- batch_nomalize는 1로 설정되어있는데, 이 말은 안 쓰겠다는 소리. 값을 높여서 정확도를 높이려고 했으나 정확도가 올라가진 않음
- subdivisions은 8인 경우 실행되지 않았다. 16으로 설정한 경우에만 실행
- cfg 값을 변경해줄때마다 anchor 값 또한 변경되었다. ==> 재설정 필요.

</div>
</details>
	
<details>
<summary>횡단보도 데이터셋을 가지고 학습을 실행.</summary>
<br>

yolov4 대신 **yolov3-tiny**를 이용하여 학습 시키니 정확도가 훨씬 높게 나타났다.
(accuracy 30%  ==> 60% 이상으로 올라갔다.)
[[횡단보도 데이터 셋 활용 출처]](https://github.com/samuelyu2002/ImVisible)

![image](https://user-images.githubusercontent.com/34594339/90633401-f138f100-e260-11ea-8d70-d78506eb1e76.png)

</div>
</details>

<details>
<summary>정확도를 올리기 위한 시행 착오</summary>
<br>

- #### 1차 시도 
	AI Hub에서 받은 데이터 셋 중 신호등이 정면에서 보이는 경우 (시각 장애인이 횡단보도 정면에 서있는 경우 신호등을 인식해야한다고 생각) 라벨링을 하였다. (약 900장)
이때 횡단보도 길이가 먼 경우를 고려하여 멀리 있는 신호등도 라벨링을 해주었다.
⇒ YoloV4를 사용하여 정확도가 30% 정도로 현저히 떨어지는 인식률이 나타났다.
- #### 2차시도
	낮은 인식률이 1차 시도에서 했던 데이터셋의 라벨링이 잘못 되었다고 판단, 좀 더 잘 정제된 횡단보도 데이터 셋을 학습시키면서 원인을 찾고자하였다. 
	횡단보도 데이터셋을 이용해 yoloV4를 이용한 학습이 여전히 낮은 인식률을 보여줬다.
	또한 cfg 설정을 바꿔보는 방법으로 학습을 시켜보았지만 별 소용이 없어, YoloV3-tiny를 사용해보았다.
	⇒ 이때 yolov4가 원인임을 발견.
- #### 3차 시도
	횡단보도가 yolov3-tiny를 이용하여 60% 이상의 인식률을 보였고, 
	신호등 데이터셋 또한 다시 라벨링하여 가까운 위치에 있는 신호등 데이터셋만 라벨링을 다시 하였다.
	
	- 새로 정제한 신호등 데이터 셋과 YoloV3-tiny를 이용하여 학습 시도
		![image](https://user-images.githubusercontent.com/34594339/90770202-61f90f80-e32c-11ea-9086-43e0d3269b24.png)

		500 여장 정도의  이미지로 50%의 인식률을 보여줬다.

	- 훈련된 위의 weight를 1차 시도의 데이터 셋까지 추가하여 학습 시도
		⇒ 48%로  정확도가 떨어졌다. 멀리 있는 신호등 사진의 데이터 셋은 오히려 인식의 정확도를 낮추는 것 같다.

- #### 4차 시도
	정확도를 더 올리기 위해  width, height를 608로 설정.
	anchor도 재정하여 실행하였으나 
	![image](https://user-images.githubusercontent.com/34594339/91044260-f676b100-e64f-11ea-81f7-50fc95d95e30.png)

	메모리 초과가 발생했다.
	⇒ batch의 크기를 조금 줄여주고, subdivision의 크기를 키워주면 된다고 함. (batch : 64, 32, 16 ...  / subdivision : 8, 16, 32, .. )
	
- #### 5차 시도
	 **batch=32 / subdivision=16으로 설정하여 재시도!**
		 ![image](https://user-images.githubusercontent.com/34594339/91061321-fe8e1b00-e666-11ea-8cfe-24373780e5ea.png)
	
	⇒ 416 크기였을 때보다 낮은 정확도,, 
	
- #### 6차 시도
	flip : 좌우 구별 감지를 이용. 정확도를 높이는 방법.
	[Data augmentation](https://nittaku.tistory.com/272)을 이용하여 정확도를 올릴수 있다고 함.</br>
	max_batches = 5200 </br>
	width, height = 416, 416 </br>
	steps=4000,4500 </br>
	
	![image](https://user-images.githubusercontent.com/34594339/91108707-aaf5ee80-e6b3-11ea-9bf6-8eeac227eb68.png)

</div>
</details>

<details>
<summary>횡단보도/신호등 탐지 모델 만들기</summary>
<br>

<details>
<summary>[08/28 ~ 08/30 학습 시도 과정]</summary>
<br>

1. 첫번째 시도
 중국데이터 + 우리 데이터 전부 : 초반에 터짐 / 아예 안됨
  cfg 설정 등을 바꿔보면서 or  데이터셋을 로컬에 다운,  구글 드라이브에 재업로드

2. 두번째 시도
신호등 원본 데이터 셋 + Bbox4는 원래 잘 됐었기 때문에 
새로 추가한 Bbox들을 하나씩 빼보면서 학습을 실행 ⇒ 25, 30을 빼고 나니 학습이 되긴함.

3. 세번째 시도
25.30 제외한 모든 데이터셋 학습 ⇒ 30분 남겨놓고 터졌다.

4. 네번째 시도
(신호등은 추가 라벨링을 한)횡단보도 원본 데이터셋만 학습 ⇒ 정확도 :: 횡단보도 / 신호등 = 58.40 / 47.38

5. 다섯번째 시도
커스텀 데이터셋만 (Bbox 전부, 25/30 여전히 안됨) : 둘다 20퍼센트대

</div>
</details>

<details>
<summary>[데이터셋을 조정해보기]</summary>
<br>

<details>
<summary>[1차 시도]</summary>
<br>

1. 횡단보도 데이터 셋 : 이미 라벨링 된 데이터 사용. 이 데이터셋의 신호등은 라벨링이 되어있지 않아 일단 사용하지 않기로 함
2. 신호등 데이터셋 : **신호등만 보이도록 이미지를 자름**
	- Bbox1(AI hub) 
	- 구글링한 신호등 데이터 
	- 직접 찍은 동영상 라벨링
3. 라벨 :  [cross walk, traffic light] ⇒ [cross walk, red light, green light, black]으로 바꿈
4. 폴더 분류 
	-  Clear(확실)
	-  neutral(중간) : 빛 번짐 없음. 형체가 확실한데 거리가 가깝고 빛번짐 살짝 허용함 (빛번짐이 심하면은 3번으로)    
	- ambiguous(애매) : 거리가 일정이상 멀어졌다고 생각이 들면 형체와 상관없이 3번 빛은 번졌는데 거리가 가깝고 박스 형체가 보이는 경우는 OK
5. 신호등 라벨링 범위
	- 어떤 신호등이든 빨간불/파란불  2칸만 라벨링
	- 화살표는 라벨링 하지 않음
	- 숫자도 라벨링 하지 않음.

	
		<image src="https://user-images.githubusercontent.com/34594339/91948589-dbd0c600-ed3a-11ea-97f5-a894caba618e.png" width="80%">


- 결과 : 횡단보도 인식은 매우 잘됨 그러나 신호등을 거의 잡지 못함 
				신호등이 매우 크게 잡힌 상태로 라벨링 되었기 때문인듯함.

</div>
</details>

### 이 이후부터는 [Data Augmentation](https://github.com/bosl95/MachineLearning_Note/tree/master/contest_preparation/data%20augmentation)에 정리

</div>
</details>

</div>
</details>

- 남은 과제들
- [x] 횡단보도 정확도 올리기 
- [x] 횡단보도 + 신호등 데이터셋을 모두 합친 학습 모델 만들기.
- [ ]  횡단보도 / 신호등 을 탐지하는 학습 모델 정확도 올리기
