import numpy as np
from paramiko import SSHClient
import iio

class objSynchSDR():
    def __init__(self, parameters):
        # SDR definitions

        self.fclk = 100e6
        self.isVerbose = parameters['isVerbose']
        self.IP = parameters['IP']

        self.hostKeysPath = parameters['hostKeysPath']
        self.username = 'root'
        self.password = 'analog'

        self.client = SSHClient()
        self.client.load_host_keys(self.hostKeysPath)
        self.client.connect(self.IP, username=self.username, password=self.password)

        self.ctx = iio.Context(('ip:'+self.IP))
        self.ctx.set_timeout(0)
        self.ctrl = self.ctx.find_device("ad9361-phy")
        self.rxLO = self.ctrl.find_channel("altvoltage0", True)
        self.txLO = self.ctrl.find_channel("altvoltage1", True)

        self.rxadc = self.ctx.find_device("cf-ad9361-lpc")
        self.rxadc.find_channel("voltage0").enabled = True
        self.rxadc.find_channel("voltage1").enabled = True
        self.rxadc.set_kernel_buffers_count=1
        
        self.txdac = self.ctx.find_device("cf-ad9361-dds-core-lpc")
        self.txdac.find_channel("voltage0",True).enabled = True
        self.txdac.find_channel("voltage1",True).enabled = True
        self.txdac.find_channel('TX1_I_F1',True).attrs['raw'].value = str(0)        # Force DAC to use DMA not DDS
        self.txdac.find_channel('TX1_Q_F1',True).attrs['raw'].value = str(0)        # Force DAC to use DMA not DDS
        self.txdac.find_channel('TX1_I_F2',True).attrs['raw'].value = str(0)        # Force DAC to use DMA not DDS
        self.txdac.find_channel('TX1_Q_F2',True).attrs['raw'].value = str(0)        # Force DAC to use DMA not DDS
        self.txdac.set_kernel_buffers_count=1 

        self.rx = self.ctrl.find_channel("voltage0")
        self.tx = self.ctrl.find_channel("voltage0",True)



    def readSDRControllerStatus(self):
        if self.isVerbose == 1:
            print("--- READ STATUS (", str(self.IP), "): ---")
        ssh_stdin, ssh_stdout, ssh_stderr = self.client.exec_command((
            "devmem 0x43C00110; "+ 
            "devmem 0x43C00114; "+
            "devmem 0x43C00118; "+
            "devmem 0x43C0011C; "+
            "devmem 0x43C00120; "+
            "devmem 0x43C00124; "+
            "devmem 0x43C00128; "+
            "devmem 0x43C0012C; "+
            "devmem 0x43C00130; ")
            )
        output = []
        for each_line in ssh_stdout:
            output.append(each_line.strip('\n'))
        print('> Wait/RX/PC/TX status: ' + str(int(output[0], 0)))
        print('> Number of detections as Mode 1: ' + str(int(output[1], 0)))
        print('> Number of detections as Mode 2: ' + str(int(output[2], 0)))
        print('> Current RX timer: ' + str(int(output[3], 0)*1000/self.fclk) + ' ms (' + output[3] + ')')
        print('> Current PC timer: ' + str(int(output[4], 0)*1000/self.fclk) + ' ms (' + output[4] + ')')
        print('> Current TX timer: ' + str(int(output[5], 0)*1000/self.fclk) + ' ms (' + output[5] + ')')
        print('> Current configuration: ' + str(int(output[6], 0)) + ' (' + output[6] + ')')
        print('> Number of single detections: ' + str(int(output[7], 0)) + ' (' + output[7] + ')')
        print('> IQ sample (16 bits for I, 16 bits for Q): ' + output[8])
        print("---")

    def setSDRControllerTimers(self,RXtimer,PCtimer,TXtimer):
        if self.isVerbose == 1:
            print("--- SET TIMERS (", str(self.IP), "): ---")

        RXtimerClk = np.ceil(self.fclk*RXtimer)
        RXtimerClk = RXtimerClk.astype('int32')
        RXtimerClkHex = hex(RXtimerClk)
        
        PCtimerClk = np.ceil(self.fclk*PCtimer)
        PCtimerClk = PCtimerClk.astype('int32')
        PCtimerClkHex = hex(PCtimerClk)
        
        TXtimerClk = np.ceil(self.fclk*TXtimer)
        TXtimerClk = TXtimerClk.astype('int32')
        TXtimerClkHex = hex(TXtimerClk)
        ssh_stdin, ssh_stdout, ssh_stderr = self.client.exec_command((
            "devmem 0x43C00100 w "+ str(RXtimerClkHex)+"; "+
            "devmem 0x43C00104 w "+ str(PCtimerClkHex)+"; "+
            "devmem 0x43C00108 w "+ str(TXtimerClkHex)
            ))
        if self.isVerbose == 1:
            print("Timers are set.")
            print("---")


    def setSDRControllerDirection(self, modeInput):
        if self.isVerbose == 1:
            print("--- SET SDR MODE (", str(self.IP), "): ---")
        ssh_stdin, ssh_stdout, ssh_stderr = self.client.exec_command(("devmem 0x43C0010C w 0x"+ str(modeInput)))
        if self.isVerbose == 1:
            print("The mode is set to " + str(modeInput) + ".")
            print("---")

    def upgradeSDR(self):
        if self.isVerbose == 1:
            print("--- UPGRADE SDR (", str(self.IP), "): ---")
            ssh_stdin, ssh_stdout, ssh_stderr = self.client.exec_command(
                'fw_setenv attr_name compatible; '+
                'fw_setenv attr_val ad9364; '+                     
                'pluto_reboot reset'
                )
        if self.isVerbose == 1:
            print("Done.")
            print("---")

    def rebootSDR(self):
        if self.isVerbose == 1:
            print("--- RESET SDR (", str(self.IP), "): ---")
        ssh_stdin, ssh_stdout, ssh_stderr = self.client.exec_command(("device_reboot reset"))
        if self.isVerbose == 1:
            print("Done.")
            print("---")


    def setTXRXcarrierFrequency(self,TXRXLO):
        self.rxLO.attrs["frequency"].value = str(int(TXRXLO))
        self.txLO.attrs["frequency"].value = str(int(TXRXLO))

        if self.isVerbose == 1:
            print("--- SET TX/RX Fc (", str(self.IP), "): ---")
            print("RX LO: ", self.rxLO.attrs["frequency"].value)
            print("TX LO: ", self.txLO.attrs["frequency"].value)
            print("---")

    def setTXattn(self,TXATTN):
        self.tx.attrs['hardwaregain'].value = str(-TXATTN)

        if self.isVerbose == 1:
            print("--- SET TX Attn (", str(self.IP), "): ---")
            print("TX Gain: ", self.tx.attrs["hardwaregain"].value)
            print("---")

    def setRXparams(self,RXLO,RXBW,RXFS,RXGAIN,RXGAINMODE):
        self.rxLO.attrs["frequency"].value = str(int(RXLO))
        self.rx.attrs["rf_bandwidth"].value = str(int(RXBW))
        self.rx.attrs["sampling_frequency"].value = str(int(RXFS))
        self.rx.attrs['gain_control_mode'].value = RXGAINMODE
        if RXGAINMODE == 'manual': # 'fast_attack', 'slow_attack', 'hybrid'
            self.rx.attrs['hardwaregain'].value = str(RXGAIN)

        if self.isVerbose == 1:
            print("--- SET RX PARAMS (", str(self.IP), "): ---")
            print("RX LO: ", self.rxLO.attrs["frequency"].value)
            print("RX BW: ", self.rx.attrs["rf_bandwidth"].value)
            print("RX FS: ", self.rx.attrs["sampling_frequency"].value)
            print("RX Gain control: ", self.rx.attrs["gain_control_mode"].value)
            print("RX Gain: ", self.rx.attrs["hardwaregain"].value)
            print("---")

        
    def setTXparams(self,TXLO,TXBW,TXFS,TXATTN):
        self.txLO.attrs["frequency"].value = str(int(TXLO))
        self.tx.attrs["rf_bandwidth"].value = str(int(TXBW))
        self.tx.attrs["sampling_frequency"].value = str(int(TXFS))
        self.tx.attrs['hardwaregain'].value = str(-TXATTN)
    
        if self.isVerbose == 1:
            print("--- SET TX PARAMS (", str(self.IP), "): ---")
            print("TX LO: ", self.txLO.attrs["frequency"].value)
            print("TX BW: ", self.tx.attrs["rf_bandwidth"].value)
            print("TX FS: ", self.tx.attrs["sampling_frequency"].value)
            print("TX Gain: ", self.tx.attrs["hardwaregain"].value)
            print("---")

    def receiveIQdata(self, numberOfSamplesAcquire):
        rxbuf = iio.Buffer(self.rxadc, numberOfSamplesAcquire, False)
        Nexecute = 1;
        for i in range(Nexecute):
            rxbuf.refill()
            iqRXbuf = rxbuf.read()
        rxbuf.cancel

        iqRX = np.frombuffer(iqRXbuf,dtype=np.int16)
        no_bits = 12
        iqRX = 2**-(no_bits-1)*iqRX.astype(np.float64)
        IQdataRX = iqRX.view(np.complex128)
        if self.isVerbose == 1:
            print("--- RECEIVE IQ DATA (", str(self.IP), "): ---")
            RXFS = self.rx.attrs["sampling_frequency"].value
            print("Payload duration: ", IQdataRX.size/int(RXFS)*1000, " ms")
            print("---")
        return IQdataRX

    def transmitIQdata(self, IQdataTX):
        inphase = (IQdataTX.real*(2**15-1))
        inphase = inphase.astype('int16')
        quadrature = (IQdataTX.imag*(2**15-1))
        quadrature = quadrature.astype('int16')

        # TX Buffer and plot
        iqTXbuf = np.empty((inphase.size + quadrature.size,), dtype=np.int16)
        iqTXbuf[0::2] = inphase
        iqTXbuf[1::2] = quadrature

        NtxbufferSize = len(inphase)
        txbuf = iio.Buffer(self.txdac, NtxbufferSize, False)
        txbuf.set_blocking_mode(False)

        # Start simulation
        iqEmpty = np.zeros((inphase.size + quadrature.size,), dtype=np.int16)

        Nexecute = 1;
        for i in range(Nexecute):
            txbuf.write(bytearray(iqTXbuf)) 
            txbuf.push()

        for i in range(3):
            txbuf.write(bytearray(iqEmpty)) 
            txbuf.push()
        txbuf.cancel

        if self.isVerbose == 1:
            print("--- TRANSMIT IQ DATA (", str(self.IP), "): ---")
            TXFS = self.tx.attrs["sampling_frequency"].value
            print("Payload duration: ", NtxbufferSize/int(TXFS)*1000, " ms")
            print("---")

