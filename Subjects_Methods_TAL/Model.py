import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import prepare_data
import create_target
import re
def preparedata(vocab,target_sentance_list):
    idxs=[vocab[w] for w in target_sentance_list]
    return torch.tensor(idxs, dtype=torch.long)
class LSTMClassification(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, category_list):
        super(LSTMClassification, self).__init__()
        self.hidden_dim = hidden_dim
        self.types_of_classes=len(category_list)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.25 )
        self.hidden2tag = nn.Linear(hidden_dim, len(category_list)+1)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))
    def forward(self,sentance):
        embeds=self.word_embeddings(sentance) #Matrice [len(Sentance)*embeddingsize]
        
        input_to_lstm=embeds.view(len(sentance),1,-1) #Resize to [len(Sentance),1,embeddingsize]
       
        lstm_out,self.hidden=self.lstm(input_to_lstm,self.hidden)
        print(lstm_out.size())
        class_score=self.hidden2tag(lstm_out.view(len(sentance),-1))
       
             
        
        #inverse class_scores [no of classes * sentance size] and then take max from each row so [no of classes * 1]


        return class_score


vocab,list_of_Category,data_list=prepare_data.prepare_data()
print(vocab)
model=LSTMClassification(25,2,len(vocab), list_of_Category)

optimizer = optim.Adam(model.parameters(), lr=0.01)

losses=[]
start=0
end=50
with open("plotdata.txt","w+", encoding="utf-8") as file:
    losses_to_print=[]
    for j in range(5):
        finalloss=0
        counter=0
        for epoch in range(20):
                for i in range(start,end):
            
                
                    remove_after_punc = re.sub("[-!,'.()`?;:]", "", data_list[i]["headline"])
                    list_of_word = remove_after_punc.split(" ")
                    print(data_list[i]["headline"],list_of_word)
                    if len(list_of_word) > 4:
                        model.zero_grad()
                        model.hidden=model.init_hidden()
                        sentance_in=preparedata(vocab,list_of_word)
                        class_scores=model(sentance_in)
                    
                        
                        target=create_target.create_target(data_list[i]["category"],list_of_Category,class_scores.size()[0])
                        target=target
                        print("Model Output",class_scores.size(),target.size(),target)
                        loss=F.cross_entropy(class_scores,target)
                        losses.append(loss.item())
                        loss.backward()
                        optimizer.step()
                        finalloss=finalloss+loss.item()
                        counter=counter+1
                start=start+50
                end=end+50
                losses.append("0101010")
        losses_to_print.append(str(finalloss/counter))
    file.write(str(losses_to_print))
    

    print(len(data_list))
    testing_loss=[]
with open("plotd_test","w+", encoding="utf-8") as file2:
    
    for i in range(16000,16300):
        remove_after_punc = re.sub("[-!,'.()`?;:]","", data_list[i]["headline"])
        list_of_word = remove_after_punc.split(" ")
        if len(list_of_word) > 2:
            sentance_in=preparedata(vocab,list_of_word)
            class_scores=model(sentance_in)
            class_scores=class_scores
            target=create_target.create_target(data_list[i]["category"],list_of_Category,class_scores.size()[0])
            
            loss2=F.cross_entropy(class_scores,target)
            class_scores=torch.transpose(class_scores,0,1)
            print(class_scores,data_list[i]["category"])
            print("testing")
            testing_loss.append((loss2.item(),data_list[i]["category"]))
    file2.write(str(testing_loss))
    print(testing_loss)
    print(list_of_Category)

