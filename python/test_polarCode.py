import numpy as np
from libraryComm import *
import time

m = 7
k = 64
polarCodeObj = objPolarCode(m,k)

EsN0dBList = np.linspace(5,10,11)
errorList = [0] * len(EsN0dBList)
Ntrial =  1000

for indSNR in range(len(EsN0dBList)):
    for indTrial in range(Ntrial):
        bitsTX = np.random.randint(2, size=k)
        codedBits = polarCodeObj.encode(bitsTX)

        IQdata = 2*codedBits-1;
        noiseVar = 10.0 ** (-EsN0dBList[indSNR]/ 10.0)
        noise =  np.sqrt(noiseVar/2) * (np.random.randn(len(IQdata)) + 1j * np.random.randn(len(IQdata)))
        IQdataNoisy = IQdata + noise
        llr_num = np.exp((-abs(IQdataNoisy.real - 1) ** 2) / noiseVar)
        llr_den = np.exp((-abs(IQdataNoisy.real - -1) ** 2) / noiseVar)
        W = np.concatenate((llr_den.reshape([1,-1]),llr_num.reshape([1,-1])),axis=0)

        t = time.time()
        bitsRX = polarCodeObj.decode(W)
        elapsed = time.time() - t
        print(elapsed)

        if all(bitsRX == bitsTX):
            print(('SNR: ' + str(EsN0dBList[indSNR]) + ' dB, PER: ' + str(errorList[indSNR]/(indTrial+1)) + ', Matched!'))
        else:
            print(('SNR: ' + str(EsN0dBList[indSNR]) + ' dB, PER: ' + str(errorList[indSNR]/(indTrial+1)) + ', Not matched...'))
            errorList[indSNR] = errorList[indSNR] + 1
print(errorList)
plt.semilogy(EsN0dBList,np.array(errorList)/Ntrial, label='TX signal')
plt.xlabel('Es/N0 [dB]')
plt.ylabel('PER')
plt.grid(True)
plt.legend()
plt.show()
aa = 1