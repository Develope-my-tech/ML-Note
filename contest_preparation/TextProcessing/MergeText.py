# cross : cross walk 라벨링 텍스트 파일이 있는 폴더
# traffic : traffic 라벨링 텍스트 파일이 있는 폴더
# new : 두 라벨을 합친 파일이 저장될 폴더
import os, shutil

CList = os.listdir('cross')
TList = os.listdir('traffic')

CL = CList.copy()
for c in CL:
    if c in TList:
        TList.remove(c)
        CList.remove(c)

        with open('cross/'+c, 'r') as cf, open('traffic/'+c, 'r') as tf:
            f = open('new/'+c, 'w')
            f.write(''.join(cf.readlines()+tf.readlines()))

for t in TList:
    shutil.copy('traffic/' +t, 'new/' + t)
for c in CList:
    shutil.copy('cross/' + c, 'new/' + c)
