from edgeServer import *
from synchSDR import *
import time

class objEdgeServerWithSDR(objEdgeServer):
    timeOffset = 0
    fcfoOffset = 0
    attnOffset = 0

    def __init__(self, radioID, Ned, signatureInputForSaving, parametersSDR):
        objEdgeServer.__init__(self, radioID, Ned, signatureInputForSaving)
        self.mySDR = objSynchSDR(parametersSDR)

        self.mySDR.setSDRControllerTimers(self.RXtimerES,self.PCtimerES,self.TXtimerES)
        self.mySDR.setSDRControllerDirection(2)
        self.mySDR.readSDRControllerStatus()

        self.mySDR.setRXparams(self.fc, self.bw, self.fs, self.gainRXES, self.gainModeES)
        self.mySDR.setTXparams(self.fc, self.bw, self.fs, self.attnTXES)

    # For emulation:
    def run(self):
        IQdataRX = 0
        while True:
            #print('The next command...')
            IQdataTX = self.step(IQdataRX)
            IQdataSDR = self.backOffPPDU*np.concatenate((self.SDRtriggerWaveform.astype(complex), IQdataTX))
            #plt.figure(1)
            #plt.plot(abs(IQdataSDR))
            #plt.show()
            self.mySDR.transmitIQdata(IQdataSDR)
            IQdataRX = self.mySDR.receiveIQdata(self.numberOfSamplesAcquire)
            time.sleep(0.5)

radioID = 0
Ned = 5

parameters = dict([
    ('isVerbose', 0), 
    ('IP', '192.168.2.1'), 
    ('hostKeysPath', 'known_hostsES'), 
    ])

signatureInputForSaving = 'emuES'
myEdgeServerWithSDR = objEdgeServerWithSDR(radioID, Ned, signatureInputForSaving, parameters)


myEdgeServerWithSDR.run()