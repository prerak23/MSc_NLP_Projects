import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
def dict():

    torch.manual_seed(1)

    data_of_new_dataset=open('new_dataset.txt','r')
    string=""
    for i in range(6000):
        string=string+data_of_new_dataset.readline()
        arr_of_string=string.split("\n")
    vocab_dict={}
    index_of_comma=0
    print(len(arr_of_string))
    for i in range(len(arr_of_string)):
        if "," in arr_of_string[i]:
            index_of_comma=arr_of_string[i].index(",")
            index_of_comma=index_of_comma+1
            string=arr_of_string[i]
            string=string[index_of_comma:]
            arr_of_string[i]=string

        for char in string:
            if char not in vocab_dict:
                vocab_dict[char]=len(vocab_dict)

    return vocab_dict
dict()
