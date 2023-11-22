from torch import Tensor, max, mean

# from torch._C import transpose
# from torch.nn import MaxPool1d, Embedding, CrossEntropyLoss
#
# # pool of size=3, stride=2
#
# m2 = MaxPool1d(2)
# m3 = MaxPool1d(3, stride=2)
# m5 = MaxPool1d(5)
# m = MaxPool1d(10)
# m_t = MaxPool1d(5)
# input = LongTensor(1, 5).random_(to=10)  # 1 line of 5 chars
#
# emb = Embedding(10, 10)
#
# embd = emb(input)
# print (input, '\n', embd)
#
# output2 = m2(embd)
# output3 = m3(embd)
# output5 = m5(embd)
#
# # print ('\n', output2, '\n', output3, '\n', output5)
#
# output = m(embd)
# print ('\n', output)
#
# # embed out
# t = transpose(embd, -1, -2)
# print (t)
# output_t = m_t(t)
# print (output_t)
#
#
# entropy = CrossEntropyLoss()

x = Tensor([[[1,2,3], [2,3,4]], [[10,20,30], [20,30,40]]])#.uniform_()
print(x)
x1= mean(x, dim=-1)
x2= max(x, dim=-2)[0]
print(x1, x1.size())
print(x2, x2.size())

print(len(x), x.size(0))

from data.corpus.corpusiterator import *

corp = CorpusIterator(CorpusIteratorArgs(False, "firsttest", 200, 10))

for sub_corp in {corp.train, corp.valid, corp.test}:
    print(sub_corp)
    i = 1
    for data, target in sub_corp():
        print(equal(data[1], target[0]), i, data.size(0))
        i+=1
