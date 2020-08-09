
## yolov4 실행해보기
1) dark net의 weight ⇒ yolov4.weights 으로 변환하는 과정.

		# Convert darknet weights to tensorflow
		## yolov4  버전
		python save_model.py --weights ./data/yolov4.weights --output ./checkpoints/yolov4-416 --input_size 416 --model yolov4 

		## yolov4-tiny 버전
		python save_model.py --weights ./data/yolov4-tiny.weights --output ./checkpoints/yolov4-tiny-416 --input_size 416 --model yolov4 --tiny
	
2) object detection이 잘 되는 지 확인하기

		# Run demo tensorflow
		python detect.py --weights ./checkpoints/yolov4-416 --size 416 --model yolov4 --image ./data/kite.jpg

		python detect.py --weights ./checkpoints/yolov4-tiny-416 --size 416 --model yolov4 --image ./data/kite.jpg --tiny

3) 결과
	![img](https://github.com/kairess/tensorflow-yolov4-tflite/raw/master/result.png)

	![image](https://user-images.githubusercontent.com/34594339/89185473-3f998f00-d5d5-11ea-99f7-45c37f85e8f0.png)

	⇒ yolov4 weight (위) / yolo4-tiny (아래)
	속도는 tiny가 훨씬 빠르다.


### 1) 데이터 가공하기
[AI Hub 데이터셋](http://www.aihub.or.kr/aidata/136)을 이용하여 신호등이 있는 사진과 Bounding Box가 되어있는 xml파일을 받았으나, **보행등과 차량등이 분류가 되어있지 않다** 는 문제점이 발생하였다.
- 해결방안
	1) 우선적으로 데이터 셋에 신호등이 다 있는 것도 아니기 때문에 1차적으로 신호등을 찾아준다.
	⇒ 신호등 label을 갖는 사진을 분류한다는 의미
	2) label을 확인함과 동시에 신호등 사진을 띄운다. 그 사진속에 있는 신호등이 보행등이라면 저장, 차량등이라면 넘어간다.
	3) 사진을 저장하는 경우에 label 데이터는 가공이 필요하다.
	현재 AI hub에서 제공되는 Bounding Box  좌표 ⇒ (좌상단 x, 좌상단 y, 우하단 x, 우하단 y) 좌표
	yolo에서 데이터 셋을 훈련시킬 때 label의 좌표 ⇒ (x, y, w, h) 좌표
			
		
		> COCO 데이터 포맷은 bouding box 값이  **x, y, w, h**  값으로 구성되어있다.
하지만 yolo 에서의 포맷은 클래스 번호와  **전체 영상 크기에 대한 center x, center y, w, h 비율 값**으로 구성된다.

		-  출처: [https://eehoeskrap.tistory.com/367](https://eehoeskrap.tistory.com/367) 
		- 참고 : [<보행등 사진만 분류하기>](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py) ++ [PILLOW와 cv2의 shape 차이]https://note.nkmk.me/en/python-opencv-pillow-image-size/
		- [https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig](https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig)
	
		내가 사용한 데이터셋은 COCO 데이터는 아니지만 이 글들을 보면서 참고해서 데이터 포맷을 맞춰줬다.

		추가)  OpenCV 좌표계
			![enter image description here](https://lh4.googleusercontent.com/ndFH6A225tFLWb7JwjyMmn539c4e1c1CmU7w4hQD6j-uO9K4diKfZ-FDr8LFuKa9oad9IaunhXRz0kD0JoRbeRV4gzUpS0ELyPKMIlpXs9FgvbJZiNGreGvWQAlMnYnRkqzo8Vlh)

		
	> ### make_label.py
	- classes : 클래스 리스트
	- cls : 데이터를 분류할 클래스 이름
	- main folder : 학습할 데이터 셋이 들어가있는 폴더.
	- ??_label.txt : labeling 값을 저장한 txt 파일
	- ??_img.txt : labeling된 이미지의 이름을 저장한 txt 파일

	- folder tree
		1) Bbox_1 ( 데이터 셋 루트 폴더)
			1) Bbox_0001
				1) 0617_01.xml
				2) img1.jpg
				3) img2.jpg
					...
			2) Bbox_0002
				...

		2) dataset ( 분류된 데이터 셋 ) ⇒ 학습에 이용할 이미지 셋
			1) Bbox_1
				1) Bbox_0001
					1) img.jpg
					2) img.txt
					3) img2.jpg
					4) img2.txt
					...
				2) Bbox_0002
				...

	- 기본 기능
		1. 원본 데이터 root 폴더의 이름 (Bbox_1)을 dataset 폴더에다가 생성.
			(이미 존재한다면 생성하지 않음)
		2.  원본 데이터 root 폴더의 하위폴더 (Bbox_0001)을 dataset/하위폴더명으로 생성.
			(마찬가지로 이미 존재한다면 생성하지 않음)
		3. 하위 폴더 탐색을 시작
			1. xml 파일을 찾음. file[0] ⇒ 가장 위에 있음.
			2. xml을 파싱, traffic light를 가지는 사진을 화면에 출력
			3. 바운딩 박스가 보행등일 경우 ``'z' 버튼`` ⇒ label.txt / img.jpg에 순서에 맞게 저장.
				**(labeling의 경우 convert 함수를 통해 데이터 포맷을 설정)**
			5. 만약 라벨링이 잘못된 경우(실수로 차량등을 저장 등..) ``'q' 버튼``을 누르면 현재 탐색하고 있는 하위 폴더를 처음부터 재탐색 (return False)
			6. 또는 라벨링을 중단하고 싶다면 'p'를 누르면 종료
			⇒ 재시작시 

					idx = len(os.listdir('dataset/' + mainfolder)) -1

				마지막으로 탐색했던 폴더로 들어가 그 폴더부터 다시 시작.
			7. 그 외에 현재 사진을 라벨링 하지 않고 계속 진행할 경우 아무 버튼이나 눌러주면 다음 사진으로 넘어감.
			
### 2) 훈련시키기 위한 설정
- #### custom data train을 위한 파일
	1) obj.data : 학습을 위한 내용이 담긴 파일
		- classes 개수
		- train.txt와 valid.txt의 경로
		- obj.names의 경로
		- weight을 저장할 폴더의 경로
	2) obj.cfg
		- 모델 구조 및 train과 관련된 설정이 들어있는 파일
		- batch 및  subdivisions 사이즈(Cuda memory 관련), width 및 height 사이즈
		- learning late, burn_in, max_batches,  policy, steps, scales 설정
		- filter : (4+1+class수) * 3
		- classes
		- anchors 및 mask 설정
	3) weight  : 미리 트레이닝 된 모델 또는 darknet53.conv.74 등의 가중치 파일
	4) obj.names : annotation에 포함되어있는 라벨링 이름 목록. 검출하고자 하는 목록
	5) images : 학습시킬 이미지들 
	6) annotation : 학습시킬 이미지에 대한 주석들
		- 각 이미지마다 주석들이 담긴 텍스트파일이 필요
			- 0001.txt
			- 0002.txt
			....
	7) train.txt : 학습시킬 이미지들의 경로들이 담긴 리스트
	8) valid.txt : 학습 시 validation 할 이미지들의 경로들이 담긴 리스트
	
- 참고자료
	- [cfg 설정을 위한 설명글](https://eehoeskrap.tistory.com/370)
	- [YOLO v4 custom데이터 훈련하기](https://keyog.tistory.com/22)
	- [커스텀 데이터 셋으로 Yolo 써 보기 1](https://jueun-park.github.io/2018-07-12/yolo-custom-dataset)
	- [https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects](https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects)

- #### Colab
	- yolo를 노트북에서도 사용하기 위해서는 GPU를 사용해야 한다. 
	- 이를 위해서 Google에서 지원하는 Colab을 이용해 yolo를 구동시킬 수 있다.
	- Colab을 세션을 12시간만 유지시켜주기 때문에 저장이 불가하다. ==> 구글 드라이브에 데이터를 저장해 놓고 마운트 해서 쓸 수 있다.

		![image](https://user-images.githubusercontent.com/34594339/89725910-db9d1d80-da4f-11ea-88bf-8ab79c47a555.png)
		
	- Colab에서 학습 시키기 위해서 Colab 안에서의 절대경로를 train.txt 파일에 지정해주어야 제대로 인식한다.. 

		![image](https://user-images.githubusercontent.com/34594339/89726951-7a7b4700-da5b-11ea-973f-da9eb26c930b.png)

	 - 실행 명령어
	
		    !./darknet detector train custom/custom.data custom/yolov4-tiny.cfg custom/yolov4-tiny.conv.29 -dont_show -map
	
	   dont_show는 원래 map과 loss에 대한 그래프가 나오는데 코랩의 리눅스 상에서는 볼 수 없는 듯하다. 에러가 나기 때문에 보지 않는 것으로 처리해준다.
