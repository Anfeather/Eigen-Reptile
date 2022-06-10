import os, sys
import shutil
import random

path = "./mini-imagenet/train"
dirs = os.listdir( path )


train_list = []
for file in dirs:
    train_list.append(file)

for i in range(len(train_list)):
    if i == 63:
        break
    file_0 = train_list[i]
    print(file_0)
    old = os.path.join(path,file_0)
    file_1 = train_list[i+1]
    new = os.path.join(path,file_1)
    pics = os.listdir( old )
    for p in pics:
        if random.random() <0.5:
            if i == 0:

                old_p = os.path.join(old,p)
                
                shutil.move(old_p,new)
            else:
                if p[:9] == train_list[i-1][:9]:
                    continue
                old_p = os.path.join(old,p)
                
                shutil.move(old_p,new)
