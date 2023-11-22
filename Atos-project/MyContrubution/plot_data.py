import matplotlib.pyplot as plt
import ast
import re
file1=open("new_filess.txt")
file2=open("new_testset_logs.txt")
string1=file1.read()
string2=file2.read()
train_data=eval(string1)
val_data=eval(string2)
counter=0
train_data_list=[]
val_data_list=[]
percen_train=[]
percen_val=[]
counter=[]
for i in range(len(train_data)):
    train_data_list.append(train_data[i][0])
    val_data_list.append(val_data[i][0])
    percen_train.append(train_data[i][1])
    percen_val.append(val_data[i][1])
    counter.append(i+1)
fig, axs=plt.subplots(2,2, figsize=(15,15))
print(train_data_list)
plt.title("Data Set Of 6000 Lines")
axs[0,0].plot(counter,train_data_list)
axs[0,0].set(title="Loss On Training Data",xlabel="epochs")
axs[0,1].plot(counter,val_data_list)
axs[0,1].set(title="Loss On Validation Data",xlabel="epochs")
axs[1,0].plot(counter,percen_train)
axs[1,0].set(title="Percentage Gain On Train Data",xlabel="epochs")

axs[1,1].set(title="Percentage Gain On Validation Data",xlabel="epochs")
axs[1,1].plot(counter,percen_val)

plt.plot()
plt.savefig("total_plot_complete.png")








