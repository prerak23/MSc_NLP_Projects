from itertools import islice
import gzip
import time
start = time.time()
def gzhand(batchsize,currentline):
    with gzip.open('/home/psrivastava/auth.txt.gz','r') as f:
        i=0
        lists=[]
        for line in islice(f,currentline,None):
            if i < batchsize :
                i+=1
                stri=str(line,'utf8').strip()
                print(stri)
                no_of_comma=stri.index(',')
                stri=stri[no_of_comma+1:]
                lists.append(stri)
            else:
                break
        end=time.time()
        print(end-start)
        print(lists)
        return lists

