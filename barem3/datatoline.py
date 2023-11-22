import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
di=open("dict.txt","r",encoding="utf8")
dicts=eval(di.readline())

coun=0
#331806 lines in the data after removing all those lines which does not give enough information
#'243':2,"320':1 is the length of the longest line but they are outliers
#19 is the length of the line which is repeadeatly repeating
#{16: 13876, 19: 131856, 111: 456, 45: 8943, 48: 561, 46: 23502, 57: 24910, 32: 25381, 37: 29974, 68: 11194, 51: 21208, 49: 14634, 27: 20129, 78: 4316, 54: 17142, 52: 19648, 40: 3002, 58: 6259, 31: 17665, 43: 5080, 59: 27098, 129: 879, 39: 15204, 30: 9351, 26: 5243, 73: 7933, 34: 12966, 50: 6431, 62: 1190, 38: 1761, 41: 9130, 53: 10795, 109: 15161, 44: 35572, 47: 28065, 36: 1910, 63: 1599, 29: 667, 56: 2024, 151: 223, 65: 5799, 64: 2422, 66: 1174, 69: 3895, 81: 793, 76: 2078, 42: 5368, 71: 6554, 70: 463, 28: 356, 25: 4787, 80: 93, 23: 243, 60: 842, 67: 82, 35: 8854, 145: 95, 75: 2500, 55: 5496, 92: 276, 87: 271, 33: 1950, 97: 400, 61: 3064, 85: 462, 77: 1100, 79: 137, 86: 7, 74: 394, 133: 17, 94: 11, 171: 7, 72: 178, 139: 39, 88: 3258, 320: 1, 102: 5, 82: 2781, 84: 1426, 89: 1128, 83: 1065, 116: 208, 115: 21, 104: 92, 99: 127, 90: 16, 107: 328, 96: 1, 22: 5, 106: 2, 242: 2, 93: 1}
def get_data_vector(currentline,batchsize):
    xt=0
    z=0
    with open("main_data_logs.txt","r",encoding="utf8") as fof:
            longtemp=torch.tensor([])

            for f in fof:
                if currentline <= z:
                    if xt < batchsize:
                       
                       strs=f
                       strs=strs.strip()
                       index=strs.find(" ")
                       templist=[dicts[strs[:index]]]

                       for x in strs[index+1:]:

                           if x in dicts:
                              templist.append(dicts[x])



                       if len(templist) < 162:
                           no=162-len(templist)

                           for y in range(no):
                              templist.append(33)

                           ss=torch.unsqueeze(torch.tensor(templist).float(),1)
                    
                           ss=ss
                                         
                           longtemp=torch.cat((longtemp,ss),1)
                        
                
                           xt+=1
                           z+=1
                       else:
                           pass
                       
                else:
                    z+=1
            print(longtemp.size(),z)
            return torch.t(longtemp)

