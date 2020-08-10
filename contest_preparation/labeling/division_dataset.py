count = 0
txt = open('all_train.txt','r')
length = len(txt.readlines()) #total line
txt.close()

i = 0
txt = open('all_train.txt','r')
f = open('custom/train.txt','w')
f2 = open('custom/validation.txt','w')

while True :
    if i == 0 :
        line = txt.readline()
        if not line :
            break
        count +=1
        if count < int(length/10)*2 :
            f2.write(line)
        else :
            f.write(line)

txt.close()
f.close()
f2.close()