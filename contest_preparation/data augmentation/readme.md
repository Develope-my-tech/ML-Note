

## Data augmentation  


<details>
<summary>참고 자료</summary>

1. https://github.com/Paperspace/DataAugmentationForObjectDetection  
2. albumentations Library 이용하기  
   https://github.com/albumentations-team/albumentations  
     
     
3. https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html  
   
 ![image](https://user-images.githubusercontent.com/34594339/91954309-96ad9380-ed3c-11ea-82f1-a83fa20af28d.png)

 </div>
</details>

- 추가할 Augmentation Dataset
1. 정사각형 사이즈의 횡단보도  데이터 (패딩)
2. 정사각형 사이즈의 신호등 데이터 (패딩)
3.  비율을 0.5로 resize한 신호등 데이터 

## transform
직사각형의 이미지를 정사각형 형태로 만들어주기
⇒ yolov3에서 416*416 형태로 학습을 진행하기 때문에 정사각형 변형을 통해 정확도 향상을 확인

'images' 폴더 대신에 들어갈 인풋 이미지 폴더 이름을 넣어줌
'output' 폴더에 정사각형 형태의 이미지가 저장됨

https://bhban.tistory.com/91



## transform2
이미지 사이즈를 일정 비율로 줄이기 ⇒ 0.5, 0.5로 비율로 줄임

![image](https://user-images.githubusercontent.com/34594339/91967657-78e92a00-ed4e-11ea-986c-71bebdead81b.png)

⇒ 이 경우는 convert 함수(꼭지점 ⇒ yolo  포맷 변환)에 shape를 전달해줄때 w, h 가 뒤바뀐다.

## check_label
transform을 여러개 해야하기 때문에 라벨링 확인이 필수
dir에 확인할 라벨링 데이터 폴더 이름을 넣어주면 라벨링된 이미지를 띄워준다.