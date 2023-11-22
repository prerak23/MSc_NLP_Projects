import gzip
import time
import io

start=time.time()

def gzhand(batchsize,currentline):
    with gzip.open("/home/psrivastava/auth.txt.gz",'r') as gz_file:
        i=0
        lists=[]
        j=0
        with io.BufferedReader(gz_file) as f:
            for line in f:
                if i > currentline:
                    if j == batchsize:
                        break
                    else:
                        stri=str(line,'utf8').strip()
                        no_of_comma=stri.index(',')
                        stri=stri[no_of_comma+1:]
                        lists.append(stri)
                        j+=1
                i+=1
            
    end=time.time()
    
    print(end-start)
    return lists

