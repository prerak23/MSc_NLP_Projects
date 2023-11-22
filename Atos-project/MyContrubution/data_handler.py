#!/usr/bin/env python3
import gzip
import numpy as np
def crap():
    with gzip.open('/home/psrivastava/auth.txt.gz', 'r') as f:
        dicr={}
        no_of_char=0
        longest_line=0
        j=0
        for i,l in enumerate(f):
            if j < 525000000 :
                print(i)
                file_line=f.readline()
                file_line=str(file_line,'utf8')
                index_of_comma=file_line.index(",")
                string=file_line[index_of_comma+1:]
                if longest_line < len(string):
                   longest_line=len(string)
                for char in string:
                    no_of_char+=1
                    if char not in dicr:
                        dicr[char]=len(dicr)
                        print("dictionary",len(dicr))
                j+=1
        np.save('dict_datass.npy',dicr)
        print(dicr,longest_line,no_of_char)
crap()
