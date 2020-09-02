
def read_anntation(file_name,img_shape):
    bounding_box_list = []

    # txt파일 읽어오기
    f = open(file_name,'r')
    labels = f.readlines()
    print("labels: ",labels)

    # shape
    dh = img_shape[0]
    dw = img_shape[1]

    for l in labels:
        info = l.split(' ')
        x = float(info[1])
        y = float(info[2])
        w = float(info[3])
        h = float(info[4])

        x1 = (2.0*x-w) / 2 * dw
        y1 = (2.0*y-h) / 2 * dh
        x2 = (2.0*x+w) / 2 * dw
        y2 = (2.0*y+h) / 2 * dh

        print(x,"/",y,"/",w,"/",h)
        print("x:", x, "/y:", y)
        print('(x1,y1): (', x1, ",", y1, ")")
        print('(x2,y2): (', x2, ",", y2, ")")

        # 한 줄씩 읽으면서 bounding box 생성
        bounding_box = [int(info[0]), x1, y1, x2, y2]
        bounding_box_list.append(bounding_box)

    # 현재 텍스트 파일에 있는 모든 bouding box, 현재 텍스트 파일명
    return bounding_box_list, file_name

from os import listdir
import cv2
import numpy as np

def read_train_dataset(dir):
    images = []
    annotations = []

    # dir 폴더의 모든 이미지 파일 읽어들임
    for file in listdir(dir):
        if 'jpg' in file.lower() or 'png' in file.lower():
            img = cv2.imread(dir + file, 1)
            images.append(img) # 컬러로 읽어들여 images 리스트에 추가
            annotation_file = file.replace(file.split('.')[-1], 'txt')
            bounding_box_list, file_name = read_anntation(dir + annotation_file, img.shape)
            annotations.append((bounding_box_list, annotation_file, file_name)) # 라벨담은 리스트, 텍스트파일명, 텍스트파일명(?)

    images = np.array(images) # Image 타입으로 변환

    return images, annotations # Image배열과 라벨배열 반환

import imgaug as ia
from imgaug import augmenters as iaa

ia.seed(1)

dir = 'images/'
images, annotations = read_train_dataset(dir)

for idx in range(len(images)):
    image = images[idx]
    boxes = annotations[idx][0] # 라벨 : [info[0], info[1], info[2], info[3], info[4]]

    ia_bounding_boxes = []
    for box in boxes:
        ia_bounding_boxes.append(ia.BoundingBox(x1=box[1], y1=box[2], x2=box[3], y2=box[4]))
    bbs = ia.BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

    seq = iaa.Sequential([
        iaa.Multiply((1.2, 1.5)),
        iaa.Affine(
            translate_px={"x":0, "y": 0},  # 시작좌표
            scale=(0.5, 0.5)
        )
    ])

    seq_det = seq.to_deterministic()

    image_aug = seq_det.augment_images([image])[0]
    # bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
    #
    # for i in range(len(bbs.bounding_boxes)):
    #     before = bbs.bounding_boxes[i]
    #     after = bbs_aug.bounding_boxes[i]
    #     print("BB %d: (%.4f, %.4f, %.4f, %.4f) -> (%.4f, %.4f, %.4f, %.4f)" % (
    #         i,
    #         before.x1, before.y1, before.x2, before.y2,
    #         after.x1, after.y1, after.x2, after.y2)
    #     )
    #
    # image_before = bbs.draw_on_image(image, thickness=5)
    # image_after = bbs_aug.draw_on_image(image_aug, thickness=5, color=[0, 0, 255])

    cv2.imshow('image_before', images[idx])
    # cv2.imshow('image_after', cv2.resize(image_after, (380, 640)))
    cv2.imshow('im', image_aug)

    cv2.waitKey(0)