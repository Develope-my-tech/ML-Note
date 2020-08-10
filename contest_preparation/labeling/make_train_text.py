import glob
import os

if __name__ == '__main__':
    folderList = os.listdir('dataset')
    for dataFolder in folderList:
        for f in os.listdir("dataset/" + dataFolder):
            text = open('all_train.txt', 'a')
            for f in glob.glob("dataset/"+dataFolder+"/"+f+"/*.jpg"):
                print("/content/drive/My Drive/Colab Notebooks/darknet/"+f.replace('\\', '/'))
                text.write("/content/drive/My Drive/Colab Notebooks/darknet/"+f.replace('\\', '/')+'\n')