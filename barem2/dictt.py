import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


#To size of the dict is 88 which includes all the characters from big to small etc etc......
#data_prepro.txt contains a dictionary which is created using Session and User Id and grouping them together

file=open("data_prepro.txt","r",encoding="utf8")
stri=file.read()
data=eval(stri)
dicts={}
for x in data:
    for y in data[x]:
        index=y.find(" ")
        if y[:index] not in dicts:
            dicts[y[:index]]=len(dicts)
        for z in y[index+1:]:
            if z not in dicts:
                dicts[z]=len(dicts)

print(dicts)
print(len(dicts))
with open("dict.txt","w",encoding="utf8") as fof:
    fof.write(str(dicts))
