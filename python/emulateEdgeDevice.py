from edgeDevice import *
from synchSDR import *

class objEdgeDeviceWithSDR(objEdgeDevice):
    def __init__(self, radioID, Ned, signatureInputForSaving, parametersSDR):
        objEdgeDevice.__init__(self, radioID, Ned, signatureInputForSaving)
        self.mySDR = objSynchSDR(parametersSDR)

        self.mySDR.setSDRControllerTimers(self.RXtimerED,self.PCtimerED,self.TXtimerED)
        self.mySDR.setSDRControllerDirection(1)
        self.mySDR.readSDRControllerStatus()

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
                print('Updating the SDR parameters...')
                print(('ED' + str(self.radioID) + ': '+ str(self.PCtimerEDCurrent)))
                self.mySDR.setTXattn(max(self.attnTXEDCurrent,0))
                self.mySDR.setSDRControllerTimers(self.RXtimerED,self.PCtimerEDCurrent,self.TXtimerED)
        
            # Receive-Process-Transmit Cycle
            print('Waiting for the next command...')
            IQdataRX = self.mySDR.receiveIQdata(self.numberOfSamplesAcquire)
            IQdataTX = self.step(IQdataRX)
            if len(IQdataTX) != 1:
                self.mySDR.transmitIQdata(self.backOffPPDU*IQdataTX)


IP = '192.168.2.2'

parametersSDR = dict([
    ('isVerbose', 0), 
    ('IP', IP), 
    ('hostKeysPath', 'known_hosts'), 
    ])

Ned = 1
signatureInputForSaving = 'emuED1'
myED = objEdgeDeviceWithSDR(0,Ned,signatureInputForSaving, parametersSDR)
myED.run()