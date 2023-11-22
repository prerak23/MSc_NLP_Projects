#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import datatoline
validation_Data=[]
def corr(out_model,target):
        maxx=torch.max(out_model,1)[1]
        corr_Char=0
        for i in range(maxx.size()[0]):
            if maxx[i].item() == target[i].item():
                corr_Char+=1
        
        return (corr_Char/maxx.size()[0])*100


def validation(model,totrainon,batch_size):
    with open("validation_log.txt","w+",encoding="utf8") as fof:
        global validation_Data
        sslines=totrainon
        lossperepoch1=0
        percen1=0
        count1=0
        uptolines=totrainon+70000
        while sslines <= (uptolines):
            data_to_train=datatoline.get_data_vector(sslines,batch_size)
            print(sslines,data_to_train.size())
            for i in range(batch_size-10): 
                start_index1=x
                target_start2=x+5
                start_index3=target_start2+1
                target_start3=start_index3+5
                
            
                out = model(data_to_train[start_index1:target_start2, :].view(5,162).cuda().long())
                out=out.view(-1,99)
                target = data_to_train[start_index3:target_start3, :].cuda().long()
                target = target.view(810)
                loss = F.cross_entropy(out, target)
                lossperepoch1 = loss.item() + lossperepoch1
                percen1 = corr(out, target)+ percen1
                count1 += 1
                print(sslines,i)
            sslines+=batch_size
        validation_Data.append(((lossperepoch1/count1),(percen1/count1)))
        fof.write(str(validation_Data))

class LSTMmodel(nn.Module):
    def __init__(self,dim_emb,vocblen,hidden_dim,maxlen):
        super(LSTMmodel,self).__init__()
        self.hidd=hidden_dim
        self.embed=nn.Embedding(vocblen,dim_emb)
        self.lstm=nn.LSTM(dim_emb,hidden_dim)
        self.hidden=self.init_hidden()
        self.predcharlayer=nn.Linear(1,1782)


    def init_hidden(self):
        return (torch.zeros(1,1,self.hidd).cuda(),
                torch.zeros(1,1,self.hidd).cuda())

    def forward(self,loglines):


        embeds=self.embed(loglines)
        
        max_out=torch.max(embeds,1)[0]
        
        max_out=torch.unsqueeze(max_out,1)
        
        lstm_out,self.hidden=self.lstm(max_out,self.hidden)
        
        lstm_out=torch.squeeze(lstm_out)
        
        lstm_out=lstm_out.view(-1,1)
        
        os=self.predcharlayer(lstm_out)
        
        return os


totrainon=300000
vocab_len=99
batch_size=2000
model=LSTMmodel(30,vocab_len,9,162)
model=model.to(device=torch.device('cuda'))
loss_func=optim.SGD(model.parameters(), lr=0.02)



with open("loss_log.txt","w+",encoding="utf8") as fof:
    lossavg=0
    percenavg=0
    ls=[]
    for epoch in range(8):
        lossperepoch=0
        count=0
        percen=0
        z=0
        torch.cuda.empty_cache()
        while z < totrainon :
            data_to_train=datatoline.get_data_vector(z,batch_size)
            
            for x in range(batch_size-10):
                start_index=x
                target_start=x+5
                start_index2=target_start+1
                target_start2=start_index2+5
                
                loss_func.zero_grad()

                model.hidden=model.init_hidden()
    
                out=model(data_to_train[start_index:target_start,:].view(5,162).cuda().long())
                print("out",out.size())
                out=out.view(-1,vocab_len)
        
                target=data_to_train[start_index2:target_start2,:].cuda().long()
        
                target=target.view(810)
                loss=F.cross_entropy(out,target)
                loss.backward()
                loss_func.step()
                lossperepoch=loss.item()+lossperepoch
                percen=corr(out,target)+percen
                
                count+=1

                
            z+=batch_size
        lossavg=(lossperepoch/count)
        percenavg=(percen/count)
        print(lossavg,percenavg,epoch)
        validation(model,totrainon,batch_size)
        ls.append((lossavg,percenavg))
    fof.write(str(ls))





