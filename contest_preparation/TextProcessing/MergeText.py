# cross : cross walk 라벨링 텍스트 파일이 있는 폴더
# traffic : traffic 라벨링 텍스트 파일이 있는 폴더
# new : 두 라벨을 합친 파일이 저장될 폴더
import os, shutil

CList = os.listdir('cross')
TList = os.listdir('traffic')

for c in CList:
    if c in TList:
        with open('cross/'+c, 'r') as cf, open('traffic/'+c, 'r') as tf:
            f = open('new/'+c, 'w')
            f.write(''.join(cf.readlines()+tf.readlines()))
    else:   # 만약에 신호등 라벨링은 있는데 횡단보도 라벨링이 없으면 그냥 신호등 라벨링 복사
        shutil.copy('traffic/'+c, 'new/'+c)