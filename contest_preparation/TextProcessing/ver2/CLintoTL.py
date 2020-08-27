import glob
import os

trafficFolder = 'Bbox_3'
crossFolder = 'box_3'
newFolder = 'new'
myExt = '*.txt' # 찾고 싶은 확장자

subfolderName = [f for f in os.listdir(trafficFolder)]

# 1. new 폴더에 하위 폴더들 미리 생성해주기
for sub in subfolderName:
    os.mkdir(newFolder+'/'+sub)

# 2. 신호등 폴더 라벨이랑 횡단보도 폴더 라벨의 각 텍스트 파일 이름 가져오기
TrafficList = [a for a in glob.glob(os.path.join(trafficFolder+"/*", myExt))]
crossList = [a for a in glob.glob(os.path.join(crossFolder+"/*", myExt))]

# 3. 합치기
for t in TrafficList:
    # 3-1. 신호등 인덱스 0 --> 1로 변환하기
    pathList = t.split('\\')
    with open(t, 'r') as traffic:
        tf_label = traffic.readline()
        tf_label = '1' +tf_label[1:]

    for c in crossList:
        if pathList[2] in c:
            with open(c, 'r') as cross:
                for line in cross.readlines():
                    tf_label += line

    with open(newFolder+'\\'+pathList[1]+'\\'+pathList[2], 'w') as newfile:
        newfile.write(tf_label)