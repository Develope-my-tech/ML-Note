  
    
> ## 01. make_labeling.py - classes : 클래스 리스트      
- cls : 데이터를 분류할 클래스 이름      
- main folder : 학습할 데이터 셋이 들어가있는 폴더.      
      
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
            2) img.txt (img.jpg의 label 값)      
            3) img2.jpg      
            4) img2.txt      
            ...      
         2) Bbox_0002      
         ...      
      
- 기본 기능      
  1. 원본 데이터 root 폴더의 이름 (Bbox_1)을 dataset 폴더에다가 생성.      
      (이미 존재한다면 생성하지 않음)      
  2. 원본 데이터 root 폴더의 하위폴더 (Bbox_0001)을 dataset/하위폴더명으로 생성.      
      (마찬가지로 이미 존재한다면 생성하지 않음)      
  3. 하위 폴더 탐색을 시작      
     1. xml 파일을 찾음. file[0] ⇒ 가장 위에 있음.      
     2. xml을 파싱, traffic light를 가지는 사진을 화면에 출력      
     3. 바운딩 박스가 보행등일 경우 ``'z' 버튼`` ⇒ label.txt / img.jpg에 순서에 맞게 저장.      
         **(labeling의 경우 convert 함수를 통해 데이터 포맷을 설정)**      
 > labeling format ==> classe index, center x, center y, ratio w, ratio h      
4. 만약 라벨링이 잘못된 경우(실수로 차량등을 저장 등..) ``'q' 버튼``을 누르면 현재 탐색하고 있는 하위 폴더를 처음부터 재탐색 (return False)      
 5. 또는 라벨링을 중단하고 싶다면 'p'를 누르면 종료      
      ⇒ 재시작시       
    
         filelist = os.listdir("dataset/"+img_path)      
         last_image_name = None      
         if filelist:      
             last_image_name = filelist[-2]      
             flag = False # 이전에 실행했던 폴더면 마지막 위치로 가기 위한 플래그    
 dataset 폴더에서 마지막으로 저장된 파일의 위치를 찾아 그 위치부터 이어서 시작할 수 있도록 함.    
     
 6. 그 외에 현재 사진을 라벨링 하지 않고 계속 진행할 경우 아무 버튼이나 눌러주면 다음 사진으로 넘어감.      
           
         
 - 실행화면        
 ![image](https://user-images.githubusercontent.com/34594339/90097605-4b254c80-dd71-11ea-9fe5-24d78e6eb917.png)      
      
      
> ## 02. make_train_text.py  
분류된 이미지 데이터 셋들의 경로를 저장할 train.txt 파일을 생성.      
이때 절대 경로로 생성을 해주었는데, 이 경로는 Google Colab에서 사용할 수 있는 절대 경로로 지정하였다.      
      
- **실행결과** : all_train.txt 파일 생성. 밑의 결과는 all_train.txt에 저장된 내용.      
![image](https://user-images.githubusercontent.com/34594339/89789461-982fd580-db5b-11ea-85a1-68c92daa20c7.png)  
  
  
### ++) **수정**   :  절대 경로 대신 상대 경로로 바꾸어주었다.  
darknet train을 실행하는 현재 경로에서 상대 경로를 찾아서 학습함으로 그에 맞게 설정해주면 된다.  
   ![image](https://user-images.githubusercontent.com/34594339/90631267-61457800-e25d-11ea-8497-53762839a6f9.png)  
  
      
> ## 03. division_dataset.py all_train.txt로 생성된 모든 이미지의 절대 경로를 8:2의 비율로 나눠, train data와 validation data로 분리한다.      
이때 custom 폴더 안에 저장하므로 실행 전 custom 폴더가 존재해야한다.      
      
- 전체 결과      
      
   ![image](https://user-images.githubusercontent.com/34594339/89789807-0ffe0000-db5c-11ea-9266-b7a23b01e7c9.png)