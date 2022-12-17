from edgeDevice import *
from synchSDR import *
import threading

class objEdgeDeviceWithSDR(objEdgeDevice):
    def __init__(self, radioID, Ned, signatureInputForSaving, parametersSDR):
        objEdgeDevice.__init__(self, radioID, Ned, signatureInputForSaving)
        self.mySDR = objSynchSDR(parametersSDR)

        self.mySDR.setSDRControllerTimers(self.RXtimerED,self.PCtimerED,self.TXtimerED)
        self.mySDR.setSDRControllerDirection(1)
        self.mySDR.readSDRControllerStatus()

        #self.mySDR.rebootSDR()

        self.mySDR.setRXparams(self.fc, self.bw, self.fs, self.gainRXED, self.gainModeED)
        self.mySDR.setTXparams(self.fc, self.bw, self.fs, self.attnTXED)
        self.fcCurrent =  self.fc
        self.PCtimerEDCurrent =  self.PCtimerED
        self.attnTXEDCurrent =  self.attnTXED
    # For emulation:
    def run(self):
        while True:
            # Update Cycle
            #print(('CFO:'+str(self.fcfoOffset)))
            self.fcCurrent = self.fcCurrent+self.fcfoOffset
            self.mySDR.setTXRXcarrierFrequency(self.fcCurrent)
            if self.isSDRUpdateRequired == True:
                self.PCtimerEDCurrent =  self.PCtimerEDCurrent + self.timeOffset
                self.attnTXEDCurrent =  max(self.attnTXEDCurrent + self.attnOffset,0)
                #print('Updating the SDR parameters...')
                #print(('ED' + str(self.radioID) + ': '+ str(self.PCtimerEDCurrent)))
                self.mySDR.setTXattn(self.attnTXEDCurrent)
                self.mySDR.setSDRControllerTimers(self.RXtimerED,self.PCtimerEDCurrent,self.TXtimerED)
        
            # Receive-Process-Transmit Cycle
            print('Waiting for the next command...')
            IQdataRX = self.mySDR.receiveIQdata(self.numberOfSamplesAcquire)
            IQdataTX = self.step(IQdataRX)
            if len(IQdataTX) != 1:
                self.mySDR.transmitIQdata(self.backOffPPDU*IQdataTX)


Ned = 5
myEdgeDevices = []
th = []
for radioID in range(Ned):
    IP = '192.168.2.' + str(radioID+1)

    parametersSDR = dict([
        ('isVerbose', 0), 
        ('IP', IP), 
        ('hostKeysPath', 'known_hosts'), 
        ])

    signatureInputForSaving = 'emuED' + str(radioID)

    myEdgeDevices.append(objEdgeDeviceWithSDR(radioID, Ned, signatureInputForSaving, parametersSDR))
    th.append(threading.Thread(target=myEdgeDevices[radioID].run))

for radioID in range(Ned):
    th[radioID].start()