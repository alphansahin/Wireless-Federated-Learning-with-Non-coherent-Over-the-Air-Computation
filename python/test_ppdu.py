import numpy as np
from libraryComm import *

ppduInst = objPPDU()
EsN0dBList = np.linspace(0,10,11)
errorList = [0] * len(EsN0dBList)
Ntrial =  1000

for indSNR in range(len(EsN0dBList)):
    for indTrial in range(Ntrial):
        bitsTX = np.random.randint(2, size=56*1)
        IQdata = ppduInst.encodePPDU(bitsTX)

        noiseVar = 10.0 ** (-EsN0dBList[indSNR]/ 10.0)
        noise =  np.sqrt(noiseVar/2) * (np.random.randn(len(IQdata)) + 1j * np.random.randn(len(IQdata)))
        IQdataNoisy = IQdata + noise
        ppduInfo = ppduInst.decodePPDU(IQdataNoisy)
        if ppduInfo['isValid'] == 0:
            errorList[indSNR] = errorList[indSNR] + 1
            print(('SNR: ' + str(EsN0dBList[indSNR]) + ' dB, PER: ' + str(errorList[indSNR]/(indTrial+1)) + ', Reason: ' + ppduInfo['reason']))
        else:
            if all(ppduInfo['dataBits'] == bitsTX):
                print(('SNR: ' + str(EsN0dBList[indSNR]) + ' dB, PER: ' + str(errorList[indSNR]/(indTrial+1))))
            else:
                print(('SNR: ' + str(EsN0dBList[indSNR]) + ' dB, PER: ' + str(errorList[indSNR]/(indTrial+1)) + ', Reason: Not matched...'))
                errorList[indSNR] = errorList[indSNR] + 1
print(errorList)
plt.semilogy(EsN0dBList,np.array(errorList)/Ntrial, label='TX signal')
plt.xlabel('Es/N0 [dB]')
plt.ylabel('PER')
plt.grid(True)
plt.legend()
plt.show()
aa = 1