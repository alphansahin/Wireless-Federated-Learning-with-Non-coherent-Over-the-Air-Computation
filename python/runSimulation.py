from edgeDevice import *
from edgeServer import *
import numpy as np


## Run simulations
Ned = 5
objEDs = []

for indED in range(Ned):
    signatureInputForSaving = 'simED' + str(indED)
    objEDs.append(objEdgeDevice(indED, Ned, signatureInputForSaving))

signatureInputForSaving = 'simES'
objES = objEdgeServer(0,Ned,signatureInputForSaving)


IQdataDownlink = np.zeros(objES.numberOfSamplesAcquire, dtype=complex) 
IQdataUplink = np.zeros(objES.numberOfSamplesAcquire, dtype=complex) 
while True:
    IQdataES = objES.step(IQdataUplink)
    IQdataDownlink = np.zeros(objES.numberOfSamplesAcquire, dtype=complex) 
    offset = objES.NsdrTail # sdr synch
    IQdataDownlink[offset:len(IQdataES)+offset] = IQdataDownlink[offset:len(IQdataES)+offset] + IQdataES

    IQdataUplink = np.zeros(objES.numberOfSamplesAcquire, dtype=complex) 
    offset = objES.deltaNtarget
    for indED in range(Ned):
        IQdataTXED = objEDs[indED].step(IQdataDownlink)
        IQdataUplink[offset:len(IQdataTXED)+offset] = IQdataUplink[offset:len(IQdataTXED)+offset] + IQdataTXED
