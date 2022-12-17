import numpy as np
from libraryComm import *

class defaults():
    def __init__(self,numberOfEDsConnected,signatureInputForSaving):
        # SDR
        self.fclk = 100e6
        self.fs = 20e6
        self.fc = 1000e6
        self.bw = 20e6

        self.RXtimerED = 50e-3
        self.PCtimerED = 750e-3
        self.TXtimerED = 50e-3 

        self.processTimeEDFast = 1.5
        self.processTimeEDSlow = 3

        self.deltaTtarget = .1e-3
        self.deltaNtarget = int(self.deltaTtarget*self.fs)
        self.Ptarget = 0.05

        self.RXtimerES = 0e-3
        self.PCtimerES = self.RXtimerED+self.PCtimerED-self.deltaTtarget
        self.TXtimerES = 0e-3

        self.gainModeES = 'manual'
        self.gainRXES = 55
        self.attnTXES = 3

        self.gainModeED = 'manual'
        self.gainRXED = 55
        self.attnTXED = 3
        
        Nsidelobes = int(20)
        Noversampling = int(2)
        rho = 0.5
        NrepeatSDRTrigger = int(4)
        Ga = np.array([1,1,1,1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,1,-1,-1,-1,-1,1,-1,1,-1])
        self.SDRtriggerWaveform = fcn_singleCarrier(rho, np.tile(Ga,NrepeatSDRTrigger), Noversampling, Nsidelobes)

        self.searchMax = int(5000)
        self.NsdrTail = int(30)
        self.numberOfEDsConnected = numberOfEDsConnected

        backOffPPDUdB = 8 # dB
        self.backOffPPDU = 10**(-backOffPPDUdB/20); # lin

        lowPAPRWaveformPowerBoostdB = 0 # dB (w.r.t backoffPPDU)
        self.lowPAPRWaveformPowerBoostdB = 10**(lowPAPRWaveformPowerBoostdB/20); # lin

        # NN
        self.isHeterogeneous = False
        self.numberOfImagesPerLabel = 500
        self.batchSizeTrain = 100
        self.learningRate = 1e-3

        self.IsSignSGDwithAbs = False
        self.signatureForSaving = signatureInputForSaving + '_heto' + str(int(self.isHeterogeneous)) + '_absenteeVote' + str(int(self.IsSignSGDwithAbs))

        # OAC
        parameters = dict([
            ('Nparameters', int(29034)), 
            ('NmaxOFDMsymbolsPerGroup', int(320)), 
            ('signatureForSaving', self.signatureForSaving), 
            ('lowPAPRWaveformPowerBoostdB', self.lowPAPRWaveformPowerBoostdB), 
            ])

        self.myNonCoherentOAC = objNonCoherentOAC(parameters)
        self.numberOfSamplesAcquire = self.myNonCoherentOAC.NmaxOFDMsymbolsPerGroup * (self.myNonCoherentOAC.Nidft+self.myNonCoherentOAC.Ncp)+10000

        # PPDU
        parameters = dict([
            ('lowPAPRWaveformPowerBoostdB', self.lowPAPRWaveformPowerBoostdB), 
            ])
        self.myPPDU = objPPDU(parameters)

        # Alignment wavefrom
        parameters = dict([
            ('numberOfSequencesToBeSearched', self.numberOfEDsConnected), 
            ('timeOffsetTarget', self.deltaNtarget), 
            ('powerTarget', self.Ptarget), 
            ('signatureForSaving', self.signatureForSaving), 
            ('lowPAPRWaveformPowerBoostdB', self.lowPAPRWaveformPowerBoostdB), 
            ])
        self.myAlignment = objAlignment(parameters)

        # Message definitions over PPDU for DL
        self.cmdTypeTriggerAlignment = int(0);
        self.cmdTypeControl = int(1);
        self.cmdTypeTriggerGradients = int(2);
        self.cmdTypeData = int(3);
        self.cmdTypeSize = int(4);

        self.maxNumberOfEDs = int(25)   # for multiple access
        self.cmdRadioIDSize = int(self.maxNumberOfEDs) # bits

        self.cmdTypeIndices = np.arange(0,self.cmdTypeSize,1, dtype=int)
        self.cmdRadioIDsIndices = self.cmdTypeSize+np.arange(0,self.cmdRadioIDSize,1, dtype=int)

        ### Offset
        self.bitsTimeOffsetsSize = int(32)
        self.bitsAttnOffsetsSize = int(8)
        self.numberOfBitsPerDevice = self.bitsTimeOffsetsSize+self.bitsAttnOffsetsSize # for multiple access

        ### Gradients
        self.maxNumberOfGroups = int(255)
        self.NgroupSize = int(np.ceil(np.log2(self.maxNumberOfGroups)))
        self.cmdIndexGroupIndices = self.cmdTypeSize+self.cmdRadioIDSize+np.arange(0,self.NgroupSize,1, dtype=int)