clear all
close all
clc

dirr = cd;
cd ..;
%addpath(genpath(['functions']))
%cd('development_Python')
%pythonModule = py.importlib.import_module('mySDRfunctions');
%py.importlib.reload(pythonModule);
cd (dirr)


timerRX = 10;
timerTX = 20;
timerPC = 80;
configuration = fi(1,0,2,0);
detectorOptionList = [4];
Noversampling = 2; % odd number
rho = 0.5;
[Ga, Gb] = fcn_golaySequenceWrapper(32, 0, 'Budisin');
sequenceOriginal = [Ga];
sequenceOversampled = repmat(sequenceOriginal,1,Noversampling).';
sequenceOversampled = sequenceOversampled(:);
sequenceXcorr = logical((sequenceOversampled+1)/2);
sequenceLength = numel(sequenceXcorr);
%[sequenceWaveform, pulseShape] = fcn_singleCarrier(rho, sequenceOriginal, Noversampling);
Nrepeat = 4;
sequenceTest = repmat(sequenceOversampled,Nrepeat,1);
[sequenceWaveform, pulseShape] = fcn_singleCarrier(rho, repmat(sequenceOriginal,Nrepeat,1), Noversampling);
%[sequenceWaveform2, pulseShape] = fcn_singleCarrier(rho, repmat(sequenceOversampled,Nrepeat,1), 1);


if(1)
    figure(10)
    plot(sequenceWaveform,'-ko')
    hold on
    grid on
    %plot(sequenceWaveform2,'-ro')
    set(gcf,'Position',[173,366,961,550]);
    drawnow
end

if(1)
    Nbefore = 200;
    Nafter = 2000;
    Nreal = 5000;
    SNRdB = 20000;

    disp(['SNR: ' num2str(SNRdB) ' dB'])

    noiseVariance = 10^(-SNRdB/10);
    waveformTest = [
        %sequenceTest; ...
        %zeros(Nbefore,1); ...
        exp(1i*2*pi*1/3)*sequenceWaveform; ...
        zeros(Nafter,1); ...
        %0.1*exp(1i*2*pi*1/3)*sequenceWaveform; ...
        %zeros(Nafter,1); ...
        %0.01*exp(1i*2*pi*1/3)*sequenceWaveform; ...
        %zeros(Nafter,1); ...
        %randn(2,1)+1i*randn(2,1);
        %zeros(Nafter,1); ...
        ];
    waveformTest = waveformTest...
        + sqrt(2*noiseVariance)/2*(randn(size(waveformTest,1),1)+1i*randn(size(waveformTest,1),1));

    waveformTest = waveformTest/max(abs(waveformTest));
    waveformTest = (int16(double(waveformTest*2^15)));
end

if(0)
    load('iqdata2450.mat')
    waveformTest = [double(I(1:200000))+1i*double(Q(1:200000))].';
end

if(0)
    pythonModule = py.importlib.import_module('mySDRfunctions');
    py.importlib.reload(pythonModule);

    NEXECUTE = 1;
    IP = ['ip:192.168.2.1'];
    RXLO =2450e6;
    RXFS = 20e6;
    RXBW = 20e6;
    RXGAIN = 70;
    NRXBUFF = 1000000;

    disp('Trying to receive...')
    tic
    [result]=pythonModule.fcn_rxOnly.feval(char(IP),...
        int64(NEXECUTE),int64(NRXBUFF),...
        int64(RXLO),int64(RXBW),int64(RXFS),...
        string(RXGAIN));
    I = int16(result{2});
    Q = int16(result{3});
    %I = double(I)/2^11;
    %Q = double(Q)/2^11;
    waveformTest = int16([double(I(:))+1i*double(Q(:))]);
    b = toc;
    disp(['> done within ' num2str(b) ' seconds.'])
    disp(' ')
end

max(real(waveformTest))
max(imag(waveformTest))

%% Simulation
input = waveformTest;
inputInPhase = (real(input));
inputQuadrature = (imag(input));
sequenceFilterToCorr = flip(sequenceOversampled);
xcorrResults = conv(sequenceFilterToCorr,input);
acorrSquareSim = conv(ones(numel(sequenceFilterToCorr),1),abs(double(input)).^2);
xcorrSquareSim = abs(xcorrResults).^2;
[peak, location] = max(xcorrSquareSim)

%% Simulin
simTime = numel(waveformTest);
slout = sim('SDRIP');
xcorrSquareHDL = getLogged(slout,'xcorrSquare');
acorrSquareHDL = getLogged(slout,'acorrSquare');

plotGroup.rhoSquare = double(xcorrSquareHDL)./double(acorrSquareHDL)/sequenceLength;
plotGroup.rhoSquare(isnan(plotGroup.rhoSquare)) = 0;
plotGroup.enableTX = getLogged(slout,'enableTX');
plotGroup.enableRX = getLogged(slout,'enableRX');
plotGroup.cntAsMode1 = getLogged(slout,'cntAsMode1');
plotGroup.cntAsMode2 = getLogged(slout,'cntAsMode2');
plotGroup.stateTimer = getLogged(slout,'stateTimer');

%% Checks
figure(1)
plot(xcorrSquareSim)
hold on
grid on
plot(xcorrSquareHDL)

figure(2)
plot(acorrSquareSim)
hold on
grid on
plot(acorrSquareHDL)

figure(3)
xx = double(xcorrSquareSim)./double(acorrSquareSim)/sequenceLength;
xx(isnan(xx))=0;
plot(xx)
hold on
grid on
plot(plotGroup.rhoSquare)

fcn_plotGroup(plotGroup,4)




