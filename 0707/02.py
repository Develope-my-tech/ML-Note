import struct

name = 'train'
maxdata = 1000
path = 'mnist/'
lbl_f = open(path + name + '-labels-idx1-ubyte', 'rb')  # 학습정답파일. 바이너리.
img_f = open(path + name + '-images-idx3-ubyte', 'rb')
csv_f = open(path + name + '.csv', 'w', encoding='utf-8')

mag, lbl_count = struct.unpack('>II', lbl_f.read(8))  # 레이블파일에서 매직넘버와 개수를 읽음
print(lbl_count)
mag, img_count = struct.unpack('>II', img_f.read(8))  # 숫자 이미지파일에서 매직넘버와 개수를 읽음
print(mag)
print(img_count)
row, col = struct.unpack('>II', img_f.read(8))  # 숫자 이미지파일에서 이미지 가로, 세로 길이 읽음
print(row)
print(col)
px = row * col  # 숫자이미지 한개의 바이트 수(크기)

res = []
for idx in range(lbl_count):
    if idx > maxdata:  # 1000이 넘으면 break
        break
    label = struct.unpack("B", lbl_f.read(1))[0]  # 정답 파일(레이블)에서 숫자 한개씩 읽음
    bdata = img_f.read(px)  # 숫자 이미지 파일에서 이미지 한 개 크기만큼 읽어서 bdata에 담음.
    sdata = list(map(lambda n: str(n), bdata))
    # print(sdata)

    csv_f.write(str(label) + ',')
    csv_f.write(','.join(sdata) + '\r\n')

    if idx < 10:  # 이 if 블럭은 써도 되고, 안써도 됨. 이미지를 단위별로 잘 불러오나 확인용
        s = 'P2 28 28 255\n'
        s += ' '.join(sdata)
        iname = path + '{0}-{1}-{2}.pgm'.format(name, idx, label)
        with open(iname, 'w', encoding='utf-8') as f:
            f.write(s)
csv_f.close()
lbl_f.close()
img_f.close()