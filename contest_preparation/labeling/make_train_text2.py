import glob
import os

if __name__ == '__main__':
    folderList = os.listdir('dataset')
    for dataFolder in folderList:
        text = open('all_train.txt', 'a')
        for file in glob.glob("dataset/"+dataFolder+"/*.jpg"):
            print("data/obj/"+dataFolder+"/"+file.split('\\')[1])
            text.write("data/obj/"+dataFolder+"/"+file.split('\\')[1]+'\n')