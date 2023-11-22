import one_hot
import dataset_creation
import torch
import numpy as np

def target(target_vector):
    final_list=[]
    for i in range(target_vector.size()[0]):
        final_list.append(target_vector[i].item())
    space_to_add=128-len(final_list)
    if space_to_add > 0:
        for i in range(space_to_add):
            final_list.append(7)
    
    final_tensor=torch.tensor(final_list).long().cuda()
    print(final_tensor.size())
    return final_tensor

