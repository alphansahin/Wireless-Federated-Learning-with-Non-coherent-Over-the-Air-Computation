import numpy as np
import matplotlib.pyplot as plt
from numpy import pi
from scipy.fftpack import fft, ifft
import scipy.io


class objDefinitions():
    def __init__(self):
        self.Nidft = int(256)
        self.Ncp = int(64)

        self.NactiveLeft = int(96)
        self.NactiveRight = int(96)
        self.NdcLeft = int(4)
        self.NdcRight = int(4)

        self.indActiveSubcarriers =np.concatenate((np.arange(-self.NactiveLeft,0,1)-self.NdcLeft, self.NdcRight+np.arange(0,self.NactiveRight,1))) % self.Nidft
        self.indTrackingActive = np.arange(0,self.NactiveRight+self.NactiveLeft,3)
        self.indDataActive = np.arange(0,self.NactiveRight+self.NactiveLeft,1)
        self.indDataActive =  np.delete(self.indDataActive,self.indTrackingActive)

        self.indTrackingSubcarriers = self.indActiveSubcarriers[self.indTrackingActive]
        self.indDataSubcarriers = self.indActiveSubcarriers[self.indDataActive]
        self.indSynchSubcarriers =np.arange(-96,97,2) % self.Nidft
        self.indNoiseSubcarriers =np.arange(-95,97,2) % self.Nidft

        self.numberOfDataSubcarriers = int(self.indDataSubcarriers.size)
        self.numberOfActiveSubcarriers = int(self.indActiveSubcarriers.size)

class objPPDU(objDefinitions):
    def __init__(self, parameters):
        objDefinitions.__init__(self)

        self.lowPAPRWaveformPowerBoostdB = parameters['lowPAPRWaveformPowerBoostdB']

        Ga = np.array([1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j])
        Gb = np.array([-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j])
        self.chestSymbols = self.lowPAPRWaveformPowerBoostdB*np.concatenate((Ga, Gb))
        self.scrambleSymbols = np.concatenate((Ga[np.arange(0,64,1)], Gb[np.arange(0,64,1)]))
        self.trackingSymbols = Ga[np.arange(0,64,1)]
        sequenceTraining = zcsequence(1,  int(97), 0)
        self.synchSymbols = np.sqrt(2)*(sequenceTraining[np.arange(self.indSynchSubcarriers.size)])

        
        self.crcLength = int(8)
        self.codeWordLength = int(128) #int(self.numberOfDataSubcarriers)
        self.messageLength = int(self.codeWordLength*0.5)-self.crcLength
        self.codeRate = self.messageLength/self.codeWordLength

        self.polarCode = objPolarCode(7,64)

        self.chestRefreshRate = int(16) # ofdm symbols
        self.maxNtotalOFDMSymbolsForData = int(100) # ofdm symbols

        self.headerSignature = [1,0,1,0,1,0,1,0]

        self.headerSignatureSize = len(self.headerSignature)
        self.headerNcodewordSize = int(np.ceil(np.log2(self.maxNtotalOFDMSymbolsForData*self.numberOfDataSubcarriers/self.codeWordLength))+1) # bits
        self.headerNpadSize = int(np.ceil(np.log2(self.messageLength))+1) # bits
        self.headerReservedSize =int(self.numberOfDataSubcarriers*self.codeRate - self.headerSignatureSize - self.headerNcodewordSize - self.headerNpadSize)

        self.headerSignatureIndices = np.arange(0,self.headerSignatureSize,1, dtype=int)
        self.headerNcodewordIndices = self.headerSignatureSize+np.arange(0,self.headerNcodewordSize,1, dtype=int)
        self.headerNpadIndices = self.headerSignatureSize+self.headerNcodewordSize+np.arange(0,self.headerNpadSize,1, dtype=int)
        self.headerReservedIndices = self.headerSignatureSize+self.headerNcodewordSize+self.headerNpadSize+np.arange(0,self.headerReservedSize,1, dtype=int)
        self.headerCRCIndices = self.headerSignatureSize+self.headerNcodewordSize+self.headerNpadSize+self.headerReservedSize+np.arange(0,self.crcLength,1, dtype=int)


    def encode(self,bitsTX):
        Nbits = int(bitsTX.size)
        NofdmData = int(np.ceil(Nbits/self.codeRate/self.numberOfDataSubcarriers))
        NchestExtra = int(np.floor((NofdmData-1)/self.chestRefreshRate))
        Ncodewords = int(np.ceil(Nbits/self.messageLength))
        Npadding = Ncodewords*self.messageLength-Nbits
        symbolsMapped = np.zeros((3+NofdmData+NchestExtra,self.Nidft), dtype=complex)

        bitsSign = np.array(self.headerSignature)
        bitsNcodeword = np.array(dec2bin(Ncodewords,self.headerNcodewordSize))
        bitsNpad = np.array(dec2bin(Npadding,self.headerNpadSize))
        bitsReserved = np.array(dec2bin(0,self.headerReservedSize))
    

        headerBitsWithoutCRC = np.concatenate((bitsSign, bitsNcodeword, bitsNpad, bitsReserved))
        headerBits = np.concatenate((headerBitsWithoutCRC, calc_crc(headerBitsWithoutCRC)))
   
    
        # Preamble
        symbolsMapped[0,self.indSynchSubcarriers] = self.synchSymbols
        symbolsMapped[1,self.indActiveSubcarriers] = self.chestSymbols
        codedBits = self.polarCode.encode(headerBits)
        symbols = 2*codedBits-1
        symbolsMapped[2,self.indDataSubcarriers] = symbols*self.scrambleSymbols
        symbolsMapped[2,self.indTrackingSubcarriers] = self.trackingSymbols

        # Payload
        dataBitsPadded = bitsTX.reshape((Nbits,-1))
        dataBitsPadded = np.concatenate((dataBitsPadded, np.zeros((Npadding,1),dtype=int)))
        dataBitsPadded = np.reshape(dataBitsPadded,(Ncodewords,-1))
        codedData = np.empty((Ncodewords,self.codeWordLength),dtype=int)

        for indCodeword in range(Ncodewords):
            codedBits = self.polarCode.encode(np.concatenate((dataBitsPadded[indCodeword,:], calc_crc(dataBitsPadded[indCodeword,:]))))
            codedData[indCodeword,:] = codedBits

        S = NofdmData+NchestExtra
        indChest = np.arange(self.chestRefreshRate,S,self.chestRefreshRate+1,dtype=int)
        indData = np.arange(0,S,1,dtype=int)
        indData = np.delete(indData, indChest)


        symbolsMapped[np.ix_(3+indData,self.indDataSubcarriers)] = (2*codedData-1)*self.scrambleSymbols
        symbolsMapped[np.ix_(3+indChest,self.indActiveSubcarriers)] = np.tile(self.chestSymbols,(NchestExtra,1)) 
        symbolsMapped[np.ix_(3+indData,self.indTrackingSubcarriers)] = np.tile(self.trackingSymbols,(NofdmData,1))

        ppdu = ifft(symbolsMapped,self.Nidft,1)*np.sqrt(self.Nidft)
        ppdu = ppdu[:,np.arange(-self.Ncp,self.Nidft)]   
        ppdu = np.reshape(ppdu,(1,-1))
        return ppdu[0]


    def decode(self,IQdataRX,fs, Npadding):
        debug = False
        if debug == True:
            plt.close('all')
            plt.figure(20)
            plt.plot(abs(IQdataRX))
            plt.grid()
            plt.show(block=False)

        L = self.Nidft//2;
        # Step 1: TO estimation
        Nchosen = self.Ncp+Npadding # Synch SDR enables time synch (64 is for CP, 35 is for SDRtriggerWaveform tails)

        # Step 2: CFO estimation
        m = np.arange(0,L,1) #[0:L-1];
        P = np.vdot(IQdataRX[Nchosen+m], IQdataRX[Nchosen+L+m])
        fcfoEst = np.angle(P)/(2*pi*L)*fs
        rCorrected = IQdataRX * np.exp(-1j*2*pi*fcfoEst*np.arange(0,IQdataRX.size,1)/fs)
        if debug == True:
            print(' > CFO...: ' + str(fcfoEst) + ' Hz'); 
            plt.figure(21)
            plt.plot(rCorrected.real)
            plt.plot(rCorrected.imag)
            plt.grid()
            plt.show(block=False)

        # Step 3: Noise Estimation
        Nsync = Nchosen-self.Ncp;
        indices = np.arange(0,(self.Nidft+self.Ncp)*1,1, dtype=int)
        rOFDMwithCP = rCorrected[Nsync+indices]
        rOFDMwithCP = np.reshape(rOFDMwithCP,(-1,self.Nidft+self.Ncp))
        rOFDMwithoutCP = rOFDMwithCP[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
        symbolsWithoutEqualization = fft(rOFDMwithoutCP)/np.sqrt(self.Nidft)
        noiseSubcarriers = symbolsWithoutEqualization[:,self.indNoiseSubcarriers]
        synchSubcarriers = symbolsWithoutEqualization[:,self.indSynchSubcarriers]

        if debug == True:
            plt.figure(22)
            plt.plot(abs(symbolsWithoutEqualization[0,:]))
            plt.grid()
            plt.show(block=False)

        noiseVarEst = np.mean(np.abs(noiseSubcarriers[0])**2)
        if noiseVarEst == 0:
            noiseVarEst = 1e-4

        signalVarEst = np.mean(np.abs(synchSubcarriers[0])**2)
        SNR = signalVarEst/noiseVarEst
        if debug == True:
            print((' > Noise var :' + str(noiseVarEst)))
            print((' > Signal var :' + str(signalVarEst)))
            print((' > SNR [dB] :' + str(10*np.log10(SNR))))

        # Step 4: Decode preamble
        indices = np.arange(0,(self.Nidft+self.Ncp)*2,1, dtype=int)
        rOFDMwithCP = rCorrected[Nsync+(self.Nidft+self.Ncp)*1+indices]
        rOFDMwithCP = np.reshape(rOFDMwithCP,(-1,self.Nidft+self.Ncp))
        rOFDMwithoutCP = rOFDMwithCP[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
        symbolsWithoutEqualization = fft(rOFDMwithoutCP)/np.sqrt(self.Nidft)
        HCFRpreamble = symbolsWithoutEqualization[0,self.indActiveSubcarriers]/self.chestSymbols

        if debug == True:
            plt.figure(23)
            plt.plot(abs(symbolsWithoutEqualization[0,:]))
            plt.grid()
            plt.show(block=False)

        Ndenoise = 32
        hCIR = fft(HCFRpreamble)
        hCIR[np.arange(Ndenoise,192-Ndenoise,1)]=0
        HCFRpreamble = ifft(hCIR)

        preambleSymbolsAfterEqualization = (symbolsWithoutEqualization[1,self.indActiveSubcarriers]*np.conjugate(HCFRpreamble)/(np.abs(HCFRpreamble)**2+1/SNR))
        
        cpe = np.mean(preambleSymbolsAfterEqualization[self.indTrackingActive]/self.trackingSymbols)
        cpe = cpe/np.abs(cpe)

        preambleDataSymbolsAfterEqualization = preambleSymbolsAfterEqualization[self.indDataActive]/self.scrambleSymbols
        preambleDataSymbolsAfterEqualization = preambleDataSymbolsAfterEqualization/cpe

        if debug == True:
            plt.figure(24)
            plt.plot(preambleDataSymbolsAfterEqualization.real,preambleDataSymbolsAfterEqualization.imag, marker = 'x', ls = '')
            plt.xlim([-2,2])
            plt.ylim([-2,2])
            plt.grid()
            plt.gca().set_aspect('equal', adjustable='box')
            plt.show(block=False)

        noiseVarPost = noiseVarEst/(noiseVarEst+np.abs(HCFRpreamble[self.indDataActive])**2)
        

        llr_num = np.exp((-abs(preambleDataSymbolsAfterEqualization.real - 1) ** 2) / noiseVarPost)
        llr_den = np.exp((-abs(preambleDataSymbolsAfterEqualization.real - -1) ** 2) / noiseVarPost)
        W = np.concatenate((llr_den.reshape([1,-1]),llr_num.reshape([1,-1])),axis=0)
        preambleMessage = self.polarCode.decode(W)

        bitsSign = preambleMessage[self.headerSignatureIndices]
        bitsNcodeword = preambleMessage[self.headerNcodewordIndices]
        bitsNpad = preambleMessage[self.headerNpadIndices]
        bitsReserved = preambleMessage[self.headerReservedIndices]
        bitsCRC = preambleMessage[self.headerCRCIndices]

        headerBitsWithoutCRC = np.concatenate((bitsSign, bitsNcodeword, bitsNpad, bitsReserved))
        bitsCRCRX = calc_crc(headerBitsWithoutCRC)


        # Step 5 : Decode payload
                
        if all(bitsCRC == bitsCRCRX):
            isValid = int(1)
            reason = None
            Ncodewords = bin2dec(preambleMessage[self.headerNcodewordIndices],0)
            Npad = bin2dec(preambleMessage[self.headerNpadIndices],0)
            Nofdm = int(np.ceil(Ncodewords*self.codeWordLength/self.numberOfDataSubcarriers))
            NchestExtra = int(np.floor((Nofdm-1)/self.chestRefreshRate))
            S = Nofdm + NchestExtra
            indChest = np.arange(self.chestRefreshRate,S,self.chestRefreshRate+1,dtype=int)
            indData = np.arange(0,S,1,dtype=int)
            indData = np.delete(indData, indChest)

            if Nsync+(self.Nidft+self.Ncp)*3 + (self.Nidft+self.Ncp)*S>len(rCorrected):
                isValid = int(0)
                dataBits = None
                reason = 'Not enough samples acquired based on the header info'
            elif Ncodewords == 0:
                isValid = int(0)
                dataBits = None
                reason = 'Number of codewords cannot be zero'
            else:
                indices = np.arange(0,(self.Nidft+self.Ncp)*S,1, dtype=int)
                rOFDMwithCP = rCorrected[Nsync+(self.Nidft+self.Ncp)*3+indices]
                rOFDMwithCP = np.reshape(rOFDMwithCP,(-1,self.Nidft+self.Ncp))
                rOFDMwithoutCP = rOFDMwithCP[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
                symbolsWithoutEqualization = fft(rOFDMwithoutCP)/np.sqrt(self.Nidft)

                if NchestExtra > 0:
                    extraChestSymbols = symbolsWithoutEqualization[indChest,:]
                    HCFRextra = extraChestSymbols[:,self.indActiveSubcarriers]/self.chestSymbols
                    HCFRextra = HCFRextra.reshape([NchestExtra, len(self.indActiveSubcarriers)])
                    HCFRextraLast = HCFRextra[-1,:].reshape([1,len(self.indActiveSubcarriers)])
                    HCFRextraFirst = HCFRextra[:-1,:].reshape([NchestExtra-1,len(self.indActiveSubcarriers)])
                    HCFRpreamble = HCFRpreamble.reshape([1,len(self.indActiveSubcarriers)])
                    Nbegin = min(self.chestRefreshRate,Nofdm)
                    Nrem = S-NchestExtra*(self.chestRefreshRate+1);
                    HCFRall = np.concatenate((np.tile(HCFRpreamble,(Nbegin,1)), np.repeat(HCFRextraFirst,self.chestRefreshRate,axis=0), np.tile(HCFRextraLast,(Nrem,1))),0)
                else:
                    HCFRall = HCFRpreamble.reshape(1,-1)

                hCIR = fft(HCFRall)
                hCIR[:,np.arange(Ndenoise,192-Ndenoise,1)]=0
                HCFRall = ifft(hCIR)

                payloadSymbols = symbolsWithoutEqualization[np.ix_(indData,self.indActiveSubcarriers)];
                payloadSymbolsAfterEqualization = payloadSymbols*np.conjugate(HCFRall)/(np.abs(HCFRall)**2+1/SNR)

                cpe = np.mean(payloadSymbolsAfterEqualization[:,self.indTrackingActive]/self.trackingSymbols,axis=1)
                cpe = cpe/np.abs(cpe)

                payloadDataSymbolsAfterEqualization = payloadSymbolsAfterEqualization[:,self.indDataActive]/self.scrambleSymbols
                payloadDataSymbolsAfterEqualization = payloadDataSymbolsAfterEqualization/cpe.reshape(-1,1)
            
                noiseVarPost = noiseVarEst/(noiseVarEst+np.abs(HCFRall[:,self.indDataActive])**2)
                llr_num = np.exp((-abs(payloadDataSymbolsAfterEqualization.real - 1) ** 2) / noiseVarPost)
                llr_den = np.exp((-abs(payloadDataSymbolsAfterEqualization.real - -1) ** 2) / noiseVarPost)
            

                messageData = np.zeros([Ncodewords,self.messageLength]);
                for indCodeword in range(Ncodewords):
                    W = np.concatenate((llr_den[indCodeword].reshape([1,-1]),llr_num[indCodeword].reshape([1,-1])),axis=0)
                    message = self.polarCode.decode(W)
                    bitsCRC = message[self.headerCRCIndices]
                    bitsCRCRX = calc_crc(message[:-self.crcLength])
                    if all(bitsCRC == bitsCRCRX):
                        messageData[indCodeword] =  message[:-self.crcLength]
                    else:
                        reason = 'One of the codewords cannot be decoded'
                        isValid = int(0)
                        break


                if isValid == 1:
                    dataBits = messageData.reshape([1,-1])
                    dataBits = dataBits[0][:-Npad or None]
                else:
                    dataBits = None
        else:
            reason = 'Invalid preamble'
            isValid = int(0)
            dataBits = None
            Ncodewords = None
            Npad = None


        ppduInfo = dict([
                ('isValid', isValid), 
                ('reason', reason), 
                ('fcfoEst', fcfoEst), 
                ('noiseVarEst', noiseVarEst), 
                ('Ncodeword', Ncodewords), 
                ('Npad', Npad), 
                ('dataBits', dataBits), 
                ])
        return ppduInfo

class objAlignment(objDefinitions):
    def __init__(self, parameters):
        objDefinitions.__init__(self)

        # Alignment parameters
        self.NsequencesToBeSearched = parameters['numberOfSequencesToBeSearched']
        self.timeOffsetTarget = parameters['timeOffsetTarget']
        self.powerTarget = parameters['powerTarget']
        self.signatureForSaving = parameters['signatureForSaving']
        self.lowPAPRWaveformPowerBoostdB = parameters['lowPAPRWaveformPowerBoostdB']

        # Alignment wavefrom
        Lsequence = int(193)
        rootList = np.arange(1,Lsequence,1)
        Nsequence = rootList.size
        sequenceTraining = np.zeros((Nsequence,Lsequence), dtype=complex)
        for indRoot in range(Nsequence):
            sequenceTraining[indRoot,:] = zcsequence(rootList[indRoot], Lsequence, 0)
        symbolsMapped = np.zeros((Nsequence,self.Nidft), dtype=complex)
        synchSymbols = (sequenceTraining[:,np.arange(self.indActiveSubcarriers.size)])
        symbolsMapped[:,self.indActiveSubcarriers] =  synchSymbols
        self.alignmentWaveform = self.lowPAPRWaveformPowerBoostdB*ifft(symbolsMapped,self.Nidft,1)*np.sqrt(self.Nidft)
        prePostPad = np.zeros((Nsequence,128),dtype=complex)
        self.alignmentWaveform =  np.hstack([prePostPad,self.alignmentWaveform,prePostPad])
        #self.alignmentWaveform = self.alignmentWaveform[:,np.arange(-self.Ncp,self.Nidft)]   
        self.lengthOfAlignmentWaveform = self.Nidft+256#+self.Ncp

        self.timeInfo = []
        self.attnInfo = []
        self.validInfo = []

    def getAlignmentWaveform(self, indexWaveform):
        prePad = np.zeros((self.lengthOfAlignmentWaveform*indexWaveform),dtype=complex)
        return np.hstack([prePad,self.alignmentWaveform[indexWaveform,:]])

    def calculateAlignmentInformation(self, IQdataRX):
        timeOffsetList = np.zeros((self.NsequencesToBeSearched), dtype=int)
        attnList = np.zeros((self.NsequencesToBeSearched), dtype=int)
        validList = np.zeros((self.NsequencesToBeSearched), dtype=int)
        rhoList = np.zeros((self.NsequencesToBeSearched))

        for indSeq in range(self.NsequencesToBeSearched):
            s = self.alignmentWaveform[indSeq,:]
            for indSample in range(len(IQdataRX)-len(s)+1):
                x = IQdataRX[indSample+np.arange(0,len(s),1,dtype = int)]
                xcorr = np.abs(np.vdot(x,s))**2
                acorr = np.vdot(x,x)*np.vdot(s,s)
                if acorr.real>0:
                    rho = xcorr/acorr.real

                    if rho > 0.25:
                        if rhoList[indSeq]<rho:
                            validList[indSeq] = 1
                            rhoList[indSeq] = rho
                            timeOffsetList[indSeq] = int(self.timeOffsetTarget+indSeq*len(s)-indSample)
                            attnList[indSeq] = int(round(10*np.log10( np.vdot(x,x).real/len(x)/self.powerTarget)))

        if True:
            print('Correlation: ' + str(rhoList))
            print('Time: ' + str(timeOffsetList) + ' samples')
            print('Attn: ' + str(attnList) + ' dB')
            print('IsValid: ' + str(validList))

        recording = True
        if recording == True:
            self.timeInfo.append([timeOffsetList])
            self.attnInfo.append([attnList])
            self.validInfo.append([validList])
            scipy.io.savemat(self.signatureForSaving + '_alignmentInfo.mat', mdict={'timeInfo': self.timeInfo, 'attnInfo': self.attnInfo, 'isValidInfo': self.validInfo})
            scipy.io.savemat(self.signatureForSaving + '_alignmentWaveform.mat', mdict={'alignmentWaveform': IQdataRX})

        return  timeOffsetList, attnList, validList, rhoList



class objNonCoherentOAC(objDefinitions):
    def __init__(self, parameters):
        objDefinitions.__init__(self)
        self.Nparameters = parameters['Nparameters']
        self.NmaxOFDMsymbolsPerGroup = parameters['NmaxOFDMsymbolsPerGroup']
        self.signatureForSaving = parameters['signatureForSaving']
        self.lowPAPRWaveformPowerBoostdB = parameters['lowPAPRWaveformPowerBoostdB']

        self.NtotalOFDMSymbols = int(np.ceil(self.Nparameters*2/(self.numberOfActiveSubcarriers)))
        self.NpaddedBins = int(self.NtotalOFDMSymbols*self.numberOfActiveSubcarriers-(self.Nparameters*2))
        self.Ngroup = int(np.ceil(self.NtotalOFDMSymbols/self.NmaxOFDMsymbolsPerGroup))
        self.NgradientsPerGroup = int(self.numberOfActiveSubcarriers/2*self.NmaxOFDMsymbolsPerGroup)
        self.majorityVote = np.zeros(self.Nparameters)

        # Sounding wavefrom
        Ga = np.array([1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j])
        Gb = np.array([-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,1 + 0j,1 + 0j,-1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,1 + 0j,0 - 1j,1 + 0j,-1 + 0j,-1 + 0j,1 + 0j,-1 + 0j,0 + 1j,-1 + 0j,1 + 0j,1 + 0j,-1 + 0j])
        self.soundingSymbols = self.lowPAPRWaveformPowerBoostdB*np.concatenate((Ga, Gb))
        
        symbolsMapped = np.zeros((1,self.Nidft), dtype=complex)
        symbolsMapped[:,self.indActiveSubcarriers] = self.soundingSymbols
        self.soundingWaveform = ifft(symbolsMapped,self.Nidft,1)*np.sqrt(self.Nidft)
        self.soundingWaveform = self.soundingWaveform[:,np.arange(-self.Ncp,self.Nidft)].reshape([-1])   
        self.lengthOfSoundingWaveform = self.Nidft+self.Ncp
        self.soundingInfo = []
        self.rxWaveformInfo = []

    def encode(self,gradients, scalar, IsSignSGDwithAbs):
        if IsSignSGDwithAbs == False:
            #gradients = (np.sign(gradients)).astype(int)
            indices = np.abs(gradients)==0
            gradients = 2*(gradients>0).astype(int)-1
            gradients[indices] = 2*np.random.randint(2,size=sum(indices.astype(int)))-1
        else:
            indices = np.abs(gradients)<0.005 # with absentees
            gradients = 2*(gradients>0).astype(int)-1
            gradients[indices] = 0
        encodedGradients = np.zeros((self.Nparameters,2))
        encodedGradients[gradients == int(-1),0]= np.sqrt(2)
        encodedGradients[gradients == int(1),1]= np.sqrt(2)
        encodedGradients = encodedGradients*np.exp(2*np.pi*1j*np.random.randint(4,size=encodedGradients.shape)/4)

        symbolsMapped = np.zeros((self.NtotalOFDMSymbols,self.Nidft), dtype=complex)
        symbolsMapped[:,self.indActiveSubcarriers] = np.concatenate((encodedGradients.reshape(-1),np.zeros(self.NpaddedBins))).reshape(-1,self.numberOfActiveSubcarriers)    
        OFDM_time = ifft(symbolsMapped,self.Nidft,1)*np.sqrt(self.Nidft)
        self.gradientWaveform = scalar*OFDM_time[:,np.arange(-self.Ncp,self.Nidft)] # scaling is due to avoid of saturation
        debug = False
        if debug == True:
            plt.close('all')
            plt.figure(20)
            plt.plot(np.abs(self.gradientWaveform.reshape([-1])))
            plt.grid()
            plt.show(block=False)

    def getGradientWaveform(self, indexGroup, numberOfDevicesForSounding, indexDevice):
        if indexGroup == self.Ngroup-1:
            indices = np.arange(np.remainder(self.NtotalOFDMSymbols, self.NmaxOFDMsymbolsPerGroup))
        else:
            indices = np.arange(self.NmaxOFDMsymbolsPerGroup)

        if numberOfDevicesForSounding > 0:
            prePad = np.zeros((self.lengthOfSoundingWaveform*indexDevice),dtype=complex)
            postPad = np.zeros((self.lengthOfSoundingWaveform*(numberOfDevicesForSounding-1-indexDevice)),dtype=complex)
            return np.hstack([prePad,self.soundingWaveform.reshape([-1]),postPad, self.gradientWaveform[indexGroup*self.NmaxOFDMsymbolsPerGroup+indices,:].reshape([-1])])
        else:
            return self.gradientWaveform[indexGroup*self.NmaxOFDMsymbolsPerGroup+indices,:].reshape([-1])

    def decode(self,rOFDMwithCP, indexGroup, Npadding, numberOfDevicesForSounding):
        debug = False
        if debug == True:
            plt.close('all')
            plt.figure(20)
            plt.plot(np.abs(rOFDMwithCP))
            plt.grid()
            plt.show(block=False)

        if numberOfDevicesForSounding > 0:
            rOFDMwithCPsounding = rOFDMwithCP[Npadding+np.arange(numberOfDevicesForSounding*self.lengthOfSoundingWaveform,dtype=int)]
            rOFDMwithCPsounding = np.reshape(rOFDMwithCPsounding,(-1,self.Nidft+self.Ncp))
            rOFDMwithoutCPsounding = rOFDMwithCPsounding[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
            symbolsWithoutEqualization = fft(rOFDMwithoutCPsounding)/np.sqrt(self.Nidft)
            symbolsWithoutEqualization[:,self.indActiveSubcarriers] = symbolsWithoutEqualization[:,self.indActiveSubcarriers]/self.soundingSymbols
            soundingInfo = np.fft.fftshift(symbolsWithoutEqualization,axes=(1,))
            plotSounding = True
            if plotSounding == True:
                plt.close('all')
                plt.figure(100)
                plt.plot(np.abs(np.transpose(soundingInfo)),marker='x')
                plt.grid()
                plt.show(block=False)

            recording = True
            if recording == True:
                self.soundingInfo.append(soundingInfo)
                scipy.io.savemat(self.signatureForSaving + '_sounding.mat', mdict={'sounding': self.soundingInfo})
                scipy.io.savemat(self.signatureForSaving + '_oacWaveform.mat', mdict={'oacWaveform': rOFDMwithCP})

            rOFDMwithCP = rOFDMwithCP[Npadding+numberOfDevicesForSounding*self.lengthOfSoundingWaveform + np.arange((self.Nidft+self.Ncp)*self.NmaxOFDMsymbolsPerGroup,dtype=int)]
        else:
            rOFDMwithCP = rOFDMwithCP[Npadding+np.arange((self.Nidft+self.Ncp)*self.NmaxOFDMsymbolsPerGroup,dtype=int)]

        rOFDMwithCP = rOFDMwithCP.reshape(-1,(self.Nidft+self.Ncp))

        if indexGroup == self.Ngroup-1:
            indices = np.arange(np.remainder(self.NtotalOFDMSymbols, self.NmaxOFDMsymbolsPerGroup),dtype=int)
            rOFDMwithCP = rOFDMwithCP[indices,:]
            rOFDMwithoutCP = rOFDMwithCP[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
            symbolsWithoutEqualization = fft(rOFDMwithoutCP)/np.sqrt(self.Nidft)
            if debug == True:
                plt.figure(21)
                plt.plot(np.abs(symbolsWithoutEqualization[0]))
                plt.plot(np.abs(symbolsWithoutEqualization[100]))
                plt.plot(np.abs(symbolsWithoutEqualization[302]))
                plt.grid()
                plt.show(block=False)

            encodedGradients = symbolsWithoutEqualization[:,self.indActiveSubcarriers]
            encodedGradients = np.abs(encodedGradients.reshape((-1,2)))
            encodedGradients = encodedGradients[:-self.NpaddedBins//2 or None,:]
        else:
            rOFDMwithoutCP = rOFDMwithCP[:,np.arange(self.Ncp,self.Nidft+self.Ncp)]
            symbolsWithoutEqualization = fft(rOFDMwithoutCP)/np.sqrt(self.Nidft)
            encodedGradients = symbolsWithoutEqualization[:,self.indActiveSubcarriers]
            encodedGradients = np.abs(encodedGradients.reshape((-1,2)))

        mv = 2*(encodedGradients[:,1]>encodedGradients[:,0]).astype(int)-1
        self.majorityVote[indexGroup*self.NgradientsPerGroup + np.arange(mv.size,dtype=int)] = mv

    def getMajorityVote(self,indexGroup):
        if indexGroup == self.Ngroup-1:
            indices = np.arange(np.remainder(self.Nparameters, self.NgradientsPerGroup))
        else:
            indices = np.arange(self.NgradientsPerGroup)
        return self.majorityVote[indexGroup*self.NgradientsPerGroup+indices]


def fcn_singleCarrier(rho, symbols, Noversampling, Nsidelobes):
    N = Noversampling * (Nsidelobes * 2 + 1)
    t = (np.arange(N) - N / 2) / Noversampling
    h_rrc = fcn_rrc(rho,t)


    symbolsUpsampled = np.zeros(Noversampling*len(symbols)-1, dtype=symbols.dtype)
    symbolsUpsampled[::Noversampling] = symbols

    waveformSC = np.convolve(symbolsUpsampled, h_rrc, mode='full')
    #plt.plot(waveformSC, label='TX signal')
    #plt.grid(True)
    #plt.legend()
    #plt.show();
    return waveformSC

def fcn_rrc(rho,t):
    h_rrc = np.zeros(t.size, dtype=np.float)

    # index for special cases
    sample_i = np.zeros(t.size, dtype=np.bool)

    # deal with special cases
    subi = t == 0
    sample_i = np.bitwise_or(sample_i, subi)
    h_rrc[subi] = 1.0 - rho + (4 * rho / np.pi)

    if rho != 0:
        subi = np.abs(t) == 1 / (4 * rho)
        sample_i = np.bitwise_or(sample_i, subi)
        h_rrc[subi] = (rho / np.sqrt(2)) \
                    * (((1 + 2 / np.pi) * (np.sin(np.pi / (4 * rho))))
                    + ((1 - 2 / np.pi) * (np.cos(np.pi / (4 * rho)))))

    # base case
    sample_i = np.bitwise_not(sample_i)
    ti = t[sample_i]
    h_rrc[sample_i] = np.sin(np.pi * ti * (1 - rho)) \
                    + 4 * rho * ti * np.cos(np.pi * ti * (1 + rho))
    h_rrc[sample_i] /= (np.pi * ti * (1 - (4 * rho * ti) ** 2))

    return h_rrc

def zcsequence(u, seq_length, q=0):
    """
    Generate a Zadoff-Chu (ZC) sequence.
    Parameters
    ----------
    u : int
        Root index of the the ZC sequence: u>0.
    seq_length : int
        Length of the sequence to be generated. Usually a prime number:
        u<seq_length, greatest-common-denominator(u,seq_length)=1.
    q : int
        Cyclic shift of the sequence (default 0).
    Returns
    -------
    zcseq : 1D ndarray of complex floats
        ZC sequence generated.
    """

    for el in [u,seq_length,q]:
        if not float(el).is_integer():
            raise ValueError('{} is not an integer'.format(el))
    if u<=0:
        raise ValueError('u is not stricly positive')
    if u>=seq_length:
        raise ValueError('u is not stricly smaller than seq_length')
    if np.gcd(u,seq_length)!=1:
        raise ValueError('the greatest common denominator of u and seq_length is not 1')

    cf = seq_length%2
    n = np.arange(seq_length)
    zcseq = np.exp( -1j * np.pi * u * n * (n+cf+2.*q) / seq_length)

    return zcseq

def dec2bin(x, numBits):
    x = np.array(x, dtype = int)
    numBits = int(numBits)
    xshape = list(x.shape)
    x = x.reshape([-1, 1])
    mask = 2**np.arange(numBits).reshape([1, numBits])
    return (x & mask).astype(bool).astype(int).reshape(xshape + [numBits])

def bin2dec(b, isSigned):
    w = b.size
    dec = np.array([np.dot(b,2**(np.arange(0,w,1)))])
    if isSigned:
        ind = np.where(dec>=2**(w-1))
        dec[ind] = dec[0] - 2**(w)
    return int(dec)


def calc_crc(bits):
    c = [1] * 8

    for b in bits:
        next_c = [0] * 8
        next_c[0] = b ^ c[7]
        next_c[1] = b ^ c[7] ^ c[0]
        next_c[2] = b ^ c[7] ^ c[1]
        next_c[3] = c[2]
        next_c[4] = c[3]
        next_c[5] = c[4]
        next_c[6] = c[5]
        next_c[7] = c[6]
        c = next_c

    return [1-b for b in c[::-1]]


class objPolarCode():

    def __init__(self, m, k):
        self.m = m
        self.k = k
        self.rate = k/2**m


        e = 1-self.k/2**self.m;
        CwBEC = (1-e)
        bitCapacityBEC = self.bitChannelCapacity(self.m, CwBEC)
        bitCapacityBEC = bitCapacityBEC[::-1]
        self.bitColumns = np.argsort(bitCapacityBEC)[::-1]
        self.bitColumns = self.bitColumns[range(self.k)]
        F = np.array([[1,0], [1, 1]])
        Fb = 1
        for n in range(self.m):
            Fb = np.kron(Fb, F)
            
        self.Fb = Fb[:,self.bitColumns]

        self.bitType = np.zeros((2**self.m),dtype=int);
        self.bitType[self.bitColumns] = int(1);
 
    def encode(self, bits):
        codedBits = np.mod(np.matmul(self.Fb,bits),2)
        return codedBits
        
    # For emulation:
    def decode(self, W):
        [output, bitsRX] = self.fcn_decoderSIC(W, self.m, self.bitType)
        return (bitsRX[self.bitColumns]-1)

    def fcn_decoderSIC(self, W, m, bitType):
        if m == 1:
            Wy1 = W[:,range(2**(m-1),2**(m))]
            Wy2 = W[:,range(0,2**(m-1))]
    
            if bitType[1] == 0: # frozen bit
                u1 = 1
            else:
                Wminus = np.zeros((Wy1.shape))
                for i in [0,1]:
                    Wminus[i] = np.sum(Wy1[::(1-2*i)]*Wy2,0)
                u1 = np.argmax(Wminus[:,0])+1

            if bitType[0] == 0:
                u2 = 1
            else:
                prob = Wy1[::(1-2*(u1-1))]*Wy2
                u2 = np.argmax(prob)+1

            output = np.array([u2, np.mod(u1+u2-2,2)+1])
            bits = np.array([u2,u1])
        else:
            Wy1 = W[:,range(2**(m-1),2**(m))]
            Wy2 = W[:,range(0,2**(m-1))]

            Wminus = np.zeros((Wy1.shape))
            for i in [0,1]:
                Wminus[i] = np.sum(Wy1[::(1-2*i)]*Wy2,0)


            [u1, bits1] = self.fcn_decoderSIC(Wminus/np.mean(Wminus), m-1, bitType[range(2**(m-1),2**(m))])

            Wplus =  np.zeros((Wy1.shape))
            for i in [0,1]:
                ind = np.where(u1==i+1)
                Wplus[:,ind] = Wy1[:,ind][::(1-2*i)]*Wy2[:,ind]

            [u2, bits2] = self.fcn_decoderSIC(Wplus/np.mean(Wplus), m-1, bitType[range(0,2**(m-1))])

            
            output = np.concatenate((u2, np.mod(u1+u2-2,2)+1))
            bits = np.concatenate((bits2, bits1))
        return output, bits


    def bitChannelCapacity(self, m, Cw):
        e = np.array(1-Cw)
        I_u1Betweeny1y2_theory = np.array(Cw*(1-e))
        I_u2Betweeny1y2u1_theory = np.array(2*Cw - I_u1Betweeny1y2_theory)

        if m == 1:
            output = np.array([I_u1Betweeny1y2_theory, I_u2Betweeny1y2u1_theory])
        else:
            output1 = self.bitChannelCapacity(m-1, I_u1Betweeny1y2_theory);
            output2 = self.bitChannelCapacity(m-1, I_u2Betweeny1y2u1_theory);
            output =  np.concatenate((output1, output2))
        return output


