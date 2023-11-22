import matplotlib.pyplot as plt
import ast
import re
file = open("plotd_test")
string = file.read()
string = re.sub("/[\[\]']","",string)
train_data=eval(string)
counter=0
output_data=[]
accuracy_data=[]
epoch=[]
epoch_value=1
for data in train_data:
    output_data.append(float(data[0]))
    accuracy_data.append(float(data[1]))
    epoch.append(counter)
    counter+=1
plt.title("loss as a function of epoch")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.plot(accuracy_data)
plt.savefig("loss_for_accuss")
print(output_data)


