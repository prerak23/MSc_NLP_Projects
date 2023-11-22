import matplotlib.pyplot as plt
import ast
import re
file = open("log_filess.txt")
string = file.read()
strings=""
for s in string:
    if "[" in s or "]" in s or "'" in s:
        strings=strings+""

    else:
        strings=strings+s
        print(strings)
print(strings)


output=strings.split(",")
counter=0
output_data=[]
epoch=[]
epoch_value=1
for data in output:
    output_data.append(float(data))
    epoch.append(counter)
    counter+=1
plt.title("loss as a function of epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(epoch,output_data)
plt.savefig("loss_as_whole")
print(output_data)
    


