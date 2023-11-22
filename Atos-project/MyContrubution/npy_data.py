import numpy as np
import gzip
import io
import os
def strt(currentline,batchsize):
    dicto=np.load('dict_datass.npy').item()
    with gzip.open("/home/psrivastava/auth.txt.gz",'r') as gz_file:
        i=0
        lists=[]
        j=0
        with io.BufferedReader(gz_file) as f:
            for line in f:
                if i >= currentline:
                    if j == batchsize:
                        break
                    else:
                        counter=0
                        
                        stri=str(line,'utf8').strip()
                        no_of_comma=stri.index(',')
                        stri=stri[no_of_comma+1:]
                        lista=np.array([])
                        for char in stri:
                             if char in dicto:
                                value_in_dicto=np.array([dicto[char]])
                                
                                lista=np.append(lista,value_in_dicto)
                                counter+=1
                                print(lista)
                        if counter>4:
                            if len(lista) < 128:
                                no_of_spaces_reuired=128-len(lista)
                                for i in range(no_of_spaces_reuired):
                                    lista=np.append(lista,[7])
                            if len(lista) == 128:
                                y=np.load("save.npy") if os.path.isfile("save.npy") else []
                                np.save("save.npy",np.append(y,lista))
                                print(y)

                        j+=1
                i+=1
strt(0,5000)

