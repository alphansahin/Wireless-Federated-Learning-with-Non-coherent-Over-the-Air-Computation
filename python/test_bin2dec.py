import numpy as np

from libraryComm import *
aa = np.array([-3,-2,2,3],dtype=int)
bb = dec2bin(aa,5)
print(bb)
aa = bin2dec(bb, 1)
print(aa)
aa = bin2dec(bb, 10)
print(aa)

aa = np.array([0,1,2,31],dtype=int)
bb = dec2bin(aa,5)
print(bb)
aa = bin2dec(bb, 1)
print(aa)
aa = bin2dec(bb, 10)
print(aa)
