

### Data augmentation  
  
1. https://github.com/Paperspace/DataAugmentationForObjectDetection  
2. albumentations Library 이용하기  
   https://github.com/albumentations-team/albumentations  
     
     
3. https://imgaug.readthedocs.io/en/latest/source/examples_bounding_boxes.html  
   
 ![image](https://user-images.githubusercontent.com/34594339/91954309-96ad9380-ed3c-11ea-82f1-a83fa20af28d.png)


## transform
직사각형의 이미지를 정사각형 형태로 만들어주기
⇒ yolov3에서 416*416 형태로 학습을 진행하기 때문에 정사각형 변형을 통해 정확도 향상을 확인

'images' 폴더 대신에 들어갈 인풋 이미지 폴더 이름을 넣어줌
'output' 폴더에 정사각형 형태의 이미지가 저장됨

https://bhban.tistory.com/91



## transform2
이미지 사이즈를 일정 비율로 줄이기 ⇒ 0.5, 0.5로 비율로 줄임

![image](https://user-images.githubusercontent.com/34594339/91967657-78e92a00-ed4e-11ea-986c-71bebdead81b.png)

