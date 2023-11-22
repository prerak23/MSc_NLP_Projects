import dataset_creation
import torch
import gz_handler
import numpy as np
from random import *
def embed(batchsize,currentline):
    total_tensor = []
    dicto=np.load('dict_datass.npy').item()

    arr_of_string=gz_handler.gzhand(batchsize,currentline)
    for string in arr_of_string:
        sentance_tensor = torch.zeros([0]).cuda()
        counter=0
        abcd=0
        for char in string:
            if char in dicto:
                value_in_dicto=dicto[char]
                current_tensor=torch.tensor([value_in_dicto]).float().cuda()
                sentance_tensor = torch.cat([sentance_tensor, current_tensor], 0)
                counter=counter+1
        if counter>4:
            xxx=len(sentance_tensor)
            if len(sentance_tensor) < 128:
                no_of_spaces_reuired=128-len(sentance_tensor)
                for i in range(no_of_spaces_reuired):
                    cs=torch.tensor([7]).float().cuda()
                    abcd+=1
                    sentance_tensor = torch.cat([sentance_tensor, cs], 0)
            


            if len(sentance_tensor) == 128:
                total_tensor.append(sentance_tensor)


    
    return total_tensor



def initia():
    return embed()

