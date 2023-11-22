import torch
def create_target(tag,category_list,length_of_input):
    one_hot_tag=torch.zeros(length_of_input).cuda()
    value=category_list[tag]
    if value == 42:
        one_hot_tag[0]=0
    else:
        one_hot_tag[0]=value
    
    return one_hot_tag.long()
