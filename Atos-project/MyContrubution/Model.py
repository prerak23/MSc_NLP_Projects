#!/usr/bin/env python3
import one_hot
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import create_target
import numpy as np


val_loss_to_log=[]

def corr(model_output,current_tensor,current_value):
    maxx=torch.max(model_output,1)[1]
    coun=0
    for i in range(current_tensor.size()[0]):
        if current_tensor[i].item() == maxx[i].item():
           coun+=1
    
    print(coun)
    return coun



def validation(model):
    with open("new_testset_logs.txt","w+",encoding="utf-8") as file2:
        global val_loss_to_log
        start=0
        total_loss_to_avg=0
        count_val=0
        corre_val=0
        validation_line_from_data=5000
        batch_size_val=550
        list_of_tensorss=one_hot.embed(batch_size_val,validation_line_from_data)
        for i in range(batch_size_val-6):
                total_loss=0
                start_point=start+i
                end_point=start_point+5
                
                sentance_tensor= torch.zeros([0])
                
                for i in range(start_point,end_point):
                    new_tensor_from_max= list_of_tensorss[i]
                    if i == start_point:
                        sentance_tensor=new_tensor_from_max
                    else:
                        sentance_tensor = torch.cat([sentance_tensor, new_tensor_from_max], -1)
                sentance_tensor=sentance_tensor.cuda()
                   
                
                sentance_tensor=sentance_tensor.view(5,128)
                out=model(sentance_tensor)
                target=create_target.target(list_of_tensorss[end_point+1])
                target=target.view(-1)
                print("target size", corre_val)
                target=target.long()

                
                loss = F.cross_entropy(out, target)
                total_loss_to_avg=loss.item()+total_loss_to_avg
                corre_val+=corr(out,target,corre_val)
                count_val+=1
        percen=corre_val/(128*count_val)
        val_loss_to_log.append((total_loss_to_avg/count_val,(percen*100)))
        file2.write(str(val_loss_to_log))



def train(batchsize,list_of_tensors,model,optimizer):
    loss_to_avg=0
    current_vall=0
    counter=0
    for iters in range(batchsize-6):
            total_loss=0
            start_point=iters
            end_point=iters+5
            for i in range(start_point,end_point):

                new_tensor_from_max= list_of_tensors[i]
            
                if i == start_point:
                    sentance_tensor=new_tensor_from_max
                else:
                    sentance_tensor = torch.cat([sentance_tensor, new_tensor_from_max], -1)
    
            
            
            sentance_tensor=sentance_tensor.view(5,128)
            optimizer.zero_grad()
            out=model(sentance_tensor).to(device=torch.device('cuda'))
            target=create_target.target(list_of_tensors[end_point+1])
            target=target.view(-1)
                  
            target=target.long().cuda()
                 
        
            loss = F.cross_entropy(out, target)
        
            loss.backward()
            optimizer.step()
            loss_to_avg=loss_to_avg+loss.item()
            counter=counter+1
            current_vall+=corr(out,target,current_vall)
    percen=current_vall/(128*counter)
    return (loss_to_avg/counter) , (percen*100)
    
 



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.embd=nn.Embedding(65,20)
    
        self.linear1=nn.Linear(100,3000)
        self.linear2=nn.Linear(3000,4000)
        self.linear3=nn.Linear(4000,8320)

    def forward(self,sentance):
        x = sentance.long().cuda()
        x = self.embd(x)
        x=torch.max(x,1)[0]
        x=x.view(1,100)
        x = F.relu(self.linear1(x)).cuda()
        x = F.relu(self.linear2(x)).cuda()
        x = self.linear3(x).cuda()
        x=x.view(128,65)
        return x

losses=[]
model=Model().to(device=torch.device('cuda'))

embedding = torch.nn.Embedding(65,20)

optimizer = optim.SGD(model.parameters(), lr=0.01)

with open("new_filess.txt","w+",encoding="utf-8") as file:
    los_to_print=[]
    counts=0
    batch_size=200
    for epoch in range(20):
        to_count_iter_on_data=0
        total_epoch_loss=0
        total_epoch_percen=0
        current_line=0
        while current_line <= 4800:
           list_of_tensor=[]
           list_of_tensor = one_hot.embed(batch_size,current_line)
           batch_loss_value,batch_percen_value=train(batch_size,list_of_tensor,model,optimizer)
           to_count_iter_on_data+=1
           total_epoch_loss=total_epoch_loss+batch_loss_value
           total_epoch_percen=total_epoch_percen+batch_percen_value
           current_line=current_line+batch_size
            
        print("to_count_iter",to_count_iter_on_data)
        total_epoch_loss=total_epoch_loss/to_count_iter_on_data
        total_epoch_percen=total_epoch_percen/to_count_iter_on_data
        los_to_print.append((total_epoch_loss,total_epoch_percen))
        validation(model)
    file.write(str(los_to_print))
    print(len(los_to_print),epoch)
print("Losses",losses)




    
 
