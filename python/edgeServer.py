from defaults import *


class objEdgeServer(defaults):
    # class attributes
    state = 0
    
    def __init__(self, radioID, numberOfEDsConnected, signatureInputForSaving):
        defaults.__init__(self,numberOfEDsConnected, signatureInputForSaving)
        self.radioID = radioID
        self.timeOffsetList = np.zeros((self.numberOfEDsConnected), dtype=int)
        self.attnList = np.zeros((self.numberOfEDsConnected), dtype=int)
        self.validList = np.zeros((self.numberOfEDsConnected), dtype=int)
        self.isAlignmentDone = np.zeros((self.numberOfEDsConnected), dtype=int)
        self.indexGroup = 0

    def step(self,IQdataRX):
        if self.state == 0:
            bitsType = np.array(dec2bin(self.cmdTypeTriggerAlignment,self.cmdTypeSize))
            bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
            bitsTX = np.concatenate((bitsType, bitsRadioIDs))
            IQdataTX = self.myPPDU.encode(bitsTX)
            print('State:' + str(self.state) + ', Trigger for alignment')
            self.state = 1
        elif self.state == 1:
            if IQdataRX.size > 1:
                debug = True
                if debug:
                    timeData = np.arange(0,len(IQdataRX[:self.searchMax]),1)/self.fs/1e-3
                    #plt.rcParams['figure.figsize'] = [20, 10]
                    plt.close('all')
                    plt.figure(100)
                    plt.plot(timeData,np.abs(IQdataRX[:self.searchMax]))
                    plt.ylim(0, 1)
                    plt.xlim(0, (len(IQdataRX[:self.searchMax])-1)/self.fs/1e-3)
                    plt.xlabel('Time (ms)')
                    plt.ylabel("Amplitude^2")
                    plt.pause(0.001)
                timeOffsetListSample, self.attnList, self.validList, _ = self.myAlignment.calculateAlignmentInformation(IQdataRX[:self.searchMax])
                self.timeOffsetList = (np.round((timeOffsetListSample/self.fs)*10**9)).astype(int)

            dataBits = np.empty((self.numberOfEDsConnected,self.numberOfBitsPerDevice),dtype=int) # Calculate alignment info for feedback
            for indED in range(self.numberOfEDsConnected):
                bitsTimeOffset = dec2bin(self.timeOffsetList[indED],self.bitsTimeOffsetsSize)
                bitsAttnOffset = dec2bin(self.attnList[indED],self.bitsAttnOffsetsSize)
                dataBits[indED,:] = np.concatenate((bitsTimeOffset, bitsAttnOffset))

            if all(self.validList==1) and all(self.timeOffsetList<4000) and all(self.isAlignmentDone==1):
                self.indexGroup = 0
                bitsType = np.array(dec2bin(self.cmdTypeTriggerGradients,self.cmdTypeSize))
                bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
                bitsIndexGroup = np.array(dec2bin(self.indexGroup,self.NgroupSize))
                bitsTX = np.concatenate((bitsType, bitsRadioIDs, bitsIndexGroup))
                IQdataTX = self.myPPDU.encode(bitsTX)
                print('State:' + str(self.state) + ', Trigger for gradient (index group ' + str(self.indexGroup) + '/' + str(self.myNonCoherentOAC.Ngroup-1) + ')')
                self.state = 2
            else:
                self.isAlignmentDone[self.validList==1] = 1
                bitsType = np.array(dec2bin(self.cmdTypeControl,self.cmdTypeSize))
                bitsRadioIDs = np.zeros((self.cmdRadioIDSize,), dtype=int)
                bitsRadioIDs[0:self.numberOfEDsConnected] = self.validList
                bitsTX = np.concatenate((bitsType, bitsRadioIDs, dataBits.flatten()))
                IQdataTX = self.myPPDU.encode(bitsTX)
                print('State:' + str(self.state) + ', Trigger for alignment')
                self.state = 0
        elif self.state == 2:
            if IQdataRX.size > 1:
                self.myNonCoherentOAC.decode(IQdataRX,self.indexGroup, self.deltaNtarget-20, self.numberOfEDsConnected)

            if self.indexGroup == self.myNonCoherentOAC.Ngroup-1:
                self.indexGroup = 0
                self.myNonCoherentOAC.encode(self.myNonCoherentOAC.majorityVote, 1, False)
                IQdataTXmv = self.myNonCoherentOAC.getGradientWaveform(self.indexGroup, 0, 0)
                #IQdataTXmv = np.reshape(IQdataTXmv,(1,-1))[0]

                bitsType = np.array(dec2bin(self.cmdTypeData,self.cmdTypeSize))
                bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
                bitsIndexGroup = np.array(dec2bin(self.indexGroup,self.NgroupSize))
                bitsTX = np.concatenate((bitsType, bitsRadioIDs, bitsIndexGroup))
                IQdataTX = self.myPPDU.encode(bitsTX)
                IQdataTX = np.concatenate((IQdataTX,IQdataTXmv))

                print('State:' + str(self.state) + ', Transmit the majority votes (index group ' + str(self.indexGroup) + '/' + str(self.myNonCoherentOAC.Ngroup-1) + ')')
                self.state = 3
            else:
                self.indexGroup = self.indexGroup + 1
                bitsType = np.array(dec2bin(self.cmdTypeTriggerGradients,self.cmdTypeSize))
                bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
                bitsIndexGroup = np.array(dec2bin(self.indexGroup,self.NgroupSize))
                bitsTX = np.concatenate((bitsType, bitsRadioIDs, bitsIndexGroup))
                IQdataTX = self.myPPDU.encode(bitsTX)
                print('State:' + str(self.state) + ', Trigger for gradient (index group ' + str(self.indexGroup) + '/' + str(self.myNonCoherentOAC.Ngroup-1) + ')')
        elif self.state == 3:
            if self.indexGroup == self.myNonCoherentOAC.Ngroup-1:
                # Trigger for alignment
                self.isAlignmentDone[:] = 0
                bitsType = np.array(dec2bin(self.cmdTypeTriggerAlignment,self.cmdTypeSize))
                bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
                bitsTX = np.concatenate((bitsType, bitsRadioIDs))
                IQdataTX = self.myPPDU.encode(bitsTX)
                print('State:' + str(self.state) + ', Trigger for alignment')
                self.state = 1
            else:
                self.indexGroup = self.indexGroup + 1
                self.myNonCoherentOAC.encode(self.myNonCoherentOAC.majorityVote, 1, False)
                IQdataTXmv = self.myNonCoherentOAC.getGradientWaveform(self.indexGroup, 0)
                IQdataTXmv = np.reshape(IQdataTXmv,(1,-1))[0]

                bitsType = np.array(dec2bin(self.cmdTypeData,self.cmdTypeSize))
                bitsRadioIDs = np.ones((self.cmdRadioIDSize,), dtype=int)
                bitsIndexGroup = np.array(dec2bin(self.indexGroup,self.NgroupSize))
                bitsTX = np.concatenate((bitsType, bitsRadioIDs, bitsIndexGroup))
                IQdataTX = self.myPPDU.encode(bitsTX)
                IQdataTX = np.concatenate((IQdataTX,IQdataTXmv))
                print('State:' + str(self.state) + ', Transmit the majority votes (index group ' + str(self.indexGroup) + '/' + str(self.myNonCoherentOAC.Ngroup-1) + ')')
        return IQdataTX