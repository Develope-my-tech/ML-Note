
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


1) 데이터 가공하기
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

		출처: [https://eehoeskrap.tistory.com/367](https://eehoeskrap.tistory.com/367) 
		참고 : [<보행등 사진만 분류하기>](https://github.com/Guanghan/darknet/blob/master/scripts/convert.py)
		[https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig](https://sites.google.com/site/bimprinciple/in-the-news/yolodibleoninggibandolopyojipaninsig)
	
		내가 사용한 데이터셋은 COCO 데이터는 아니지만 이 글들을 보면서 참고해서 데이터 포맷을 맞춰줬다.

		추가)  OpenCV 좌표계
			![enter image description here](https://lh4.googleusercontent.com/ndFH6A225tFLWb7JwjyMmn539c4e1c1CmU7w4hQD6j-uO9K4diKfZ-FDr8LFuKa9oad9IaunhXRz0kD0JoRbeRV4gzUpS0ELyPKMIlpXs9FgvbJZiNGreGvWQAlMnYnRkqzo8Vlh)
		
2) 훈련시키기 위한 설정
[https://keyog.tistory.com/22](https://keyog.tistory.com/22)
