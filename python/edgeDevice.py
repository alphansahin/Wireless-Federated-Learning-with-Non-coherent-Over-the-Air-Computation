from defaults import *
from libraryLearning import *
import matplotlib.pyplot as plt

class objEdgeDevice(defaults):
    # class attributes
    state = 0
    isSDRUpdateRequired = True
    timeOffset = 0
    fcfoOffset = 0
    attnOffset = 0
    
    def __init__(self, radioID, numberOfEDsConnected, signatureInputForSaving):
        defaults.__init__(self,numberOfEDsConnected, signatureInputForSaving)
        self.radioID = radioID
        if self.isHeterogeneous == True:
            parameters = dict([
                ('labels', np.array([0, 1, 2, 3, 4, 5],dtype=int)+np.array(self.radioID)), 
                ('indexShard', self.radioID), 
                ('numberOfImagesPerLabel', self.numberOfImagesPerLabel), 
                ])
        else:
            parameters = dict([
                ('labels', np.arange(10)), 
                ('indexShard', self.radioID), 
                ('numberOfImagesPerLabel',  self.numberOfImagesPerLabel), 
                ])
        imagesTraining, labelsTraining = fcn_dataLoader(parameters)
        parameters = dict([
            ('batchSizeTrain', self.batchSizeTrain), 
            ('learningRate', self.learningRate), 
            ('imagesTraining', imagesTraining), 
            ('labelsTraining', labelsTraining), 
            ('signatureForSaving', self.signatureForSaving), 
            ])
        self.mylearningEngine = objLearningEngine(parameters)
        self.majorityVotes = np.zeros((self.mylearningEngine.numberOfParameters))
        self.myNonCoherentOAC.encode(self.mylearningEngine.step(self.majorityVotes), np.sqrt(1/self.numberOfEDsConnected), self.IsSignSGDwithAbs)
        


    def step(self,IQdataRX):
        if IQdataRX.size > 50:
            self.isSDRUpdateRequired = False
            ppduInfo = self.myPPDU.decode(IQdataRX,self.fs,self.NsdrTail)
            if ppduInfo['isValid'] == 1:
                cmdType = bin2dec(ppduInfo['dataBits'][self.cmdTypeIndices],0)
                radioIDs = ppduInfo['dataBits'][self.cmdRadioIDsIndices]
                self.fcfoOffset = ppduInfo['fcfoEst']
                #print(str(cmdType))
                if cmdType == self.cmdTypeTriggerAlignment:
                    if radioIDs[self.radioID] == 1:
                        IQdataTX = self.myAlignment.getAlignmentWaveform(self.radioID)
                    else:
                        IQdataTX = np.array([0]) 
                elif cmdType == self.cmdTypeTriggerGradients:
                    if radioIDs[self.radioID] == 1:
                        indexGroup = bin2dec(ppduInfo['dataBits'][self.cmdIndexGroupIndices],0)
                        IQdataTX = self.myNonCoherentOAC.getGradientWaveform(indexGroup, self.numberOfEDsConnected, self.radioID) #np.reshape(IQdataTX,(1,-1))[0]
                    else:
                        IQdataTX = np.array([0]) 
                elif cmdType == self.cmdTypeControl:
                    if radioIDs[self.radioID] == 1:
                        Noffset = self.cmdTypeSize + self.cmdRadioIDSize + self.numberOfBitsPerDevice*self.radioID
                        timeOffsetIndices = Noffset + np.arange(0,self.bitsTimeOffsetsSize,1)
                        attnIndices = Noffset + self.bitsTimeOffsetsSize + np.arange(0,self.bitsAttnOffsetsSize,1)
                        self.timeOffset = bin2dec(ppduInfo['dataBits'][timeOffsetIndices],1)*1e-9
                        self.attnOffset = bin2dec(ppduInfo['dataBits'][attnIndices],1)
                        
                        self.isSDRUpdateRequired = True
                    IQdataTX = np.array([0]) 
                elif cmdType == self.cmdTypeData:
                    if radioIDs[self.radioID] == 1:
                        indexGroup = bin2dec(ppduInfo['dataBits'][self.cmdIndexGroupIndices],0)
                        self.myNonCoherentOAC.decode(IQdataRX, indexGroup, self.NsdrTail+4*(self.myNonCoherentOAC.Nidft+self.myNonCoherentOAC.Ncp), 0)

                        if indexGroup == self.myNonCoherentOAC.Ngroup-1:
                            self.myNonCoherentOAC.encode(self.mylearningEngine.step(self.myNonCoherentOAC.majorityVote), np.sqrt(1/self.numberOfBitsPerDevice), self.IsSignSGDwithAbs)
                            self.myNonCoherentOAC.majorityVote[:] = 0

                    IQdataTX = np.array([0]) 
                else:
                    IQdataTX = np.array([0]) 
            else:
                IQdataTX = np.array([0]) 
        else:
            IQdataTX = np.array([0]) 

        return IQdataTX