/*
 * File Name:         hdl_prj\ipcore\SDRIPDUT_ip_v1_0\include\SDRIPDUT_ip_addr.h
 * Description:       C Header File
 * Created:           2022-06-06 21:53:53
*/

#ifndef SDRIPDUT_IP_H_
#define SDRIPDUT_IP_H_

#define  IPCore_Reset_SDRIPDUT_ip                0x0  //write 0x1 to bit 0 to reset IP core
#define  IPCore_Enable_SDRIPDUT_ip               0x4  //enabled (by default) when bit 0 is 0x1
#define  IPCore_Timestamp_SDRIPDUT_ip            0x8  //contains unique IP timestamp (yymmddHHMM): 2206062153
#define  timerRX_Data_SDRIPDUT_ip                0x100  //data register for Inport timerRX
#define  timerPC_Data_SDRIPDUT_ip                0x104  //data register for Inport timerPC
#define  timerTX_Data_SDRIPDUT_ip                0x108  //data register for Inport timerTX
#define  configuration_Data_SDRIPDUT_ip          0x10C  //data register for Inport configuration
#define  stateTimer_Data_SDRIPDUT_ip             0x110  //data register for Outport stateTimer
#define  cntDetectionAsMode1_Data_SDRIPDUT_ip    0x114  //data register for Outport cntDetectionAsMode1
#define  cntDetectionAsMode2_Data_SDRIPDUT_ip    0x118  //data register for Outport cntDetectionAsMode2
#define  timerRXCurrent_Data_SDRIPDUT_ip         0x11C  //data register for Outport timerRXCurrent
#define  timerPCCurrent_Data_SDRIPDUT_ip         0x120  //data register for Outport timerPCCurrent
#define  timerTXCurrent_Data_SDRIPDUT_ip         0x124  //data register for Outport timerTXCurrent
#define  configurationCurrent_Data_SDRIPDUT_ip   0x128  //data register for Outport configurationCurrent
#define  cntDetectionSingle_Data_SDRIPDUT_ip     0x12C  //data register for Outport cntDetectionSingle
#define  IQdataSample_Data_SDRIPDUT_ip           0x130  //data register for Outport IQdataSample

#endif /* SDRIPDUT_IP_H_ */
