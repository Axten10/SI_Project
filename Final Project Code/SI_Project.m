%% Load Data Set
data2_13 = load("2_13_24_HFS_data.mat");
data2_16 = load("2_16_24_HFS_data.mat");
data2_24 = load("2_24_24_HFS_data.mat");
data2_27 = load("2_27_24_HFS_data.mat");

%% Define Variables
% Test 1
sddata1 = data2_13.SDData;
Temperature1 = sddata1(:,2);
HFS1 = sddata1(:,4)*(5/1023);

% Test 2
sddata2 = data2_16.SDData_1;
Temperature2 = sddata2(:,2);
HFS2 = sddata2(:,4)*(5/1023);

% Test 3
sddata3 = data2_24.SDData_2;
Temperature3 = sddata3(:,2);
HFS3 = sddata3(:,4)*(5/1023);

% Test 4
sddata4 = data2_27.SDData_3;
Temperature4 = sddata4(:,3);
HFS4 = sddata4(:,5)*(5/1023);

% Parsing Unusable Data/ Adjusting for proper time 

Temperature1 = Temperature1(58:436,:);
HFS1 = HFS1(58:436,:);

% Temperature2 = Temperature2(58:436,:);
% HFS2 = HFS2(58:436,:);
% 
% Temperature3 = Temperature3(58:436,:);
% HFS3 = HFS3(58:436,:);
% 
% Temperature4 = Temperature4(58:436,:);
% HFS4 = HFS4(58:436,:);


%% Creating Time Vectors for each Experiment
Ts = 1;

% Test 1
N1 = length(Temperature1);
t1 = Ts:Ts:N1*Ts;
% Test 2 
N2 = length(Temperature2);
t2 = Ts:Ts:N2*Ts;
% Test 3
N3 = length(Temperature3);
t3 = Ts:Ts:N3*Ts;
% Test 4
N4 = length(Temperature4);
t4 = Ts:Ts:N4*Ts;



%% Plotting data sets
%Plotting test 1
figure(1)
subplot(2, 1, 1)
plot(t1,Temperature1)
title("Internal Room Temperature vs Time")
xlabel("Time (min)")
ylabel("Temperature (deg C)")

subplot(2,1,2)
plot(t1, HFS1)
title("Heat Flux vs Time")
xlabel("Time (min)")
ylabel("Voltage (V)")

%Plotting test 2
figure(2)
subplot(2, 1, 1)
plot(t2,Temperature2)
title("Internal Room Temperature vs Time")
xlabel("Time (min)")
ylabel("Temperature (deg C)")

subplot(2,1,2)
plot(t2, HFS2)
title("Heat Flux vs Time")
xlabel("Time (min)")
ylabel("Voltage (V)")

%Plotting test 3
figure(3)
subplot(2, 1, 1)
plot(t3,Temperature3)
title("Internal Room Temperature vs Time")
xlabel("Time (min)")
ylabel("Temperature (deg C)")

subplot(2,1,2)
plot(t3, HFS3)
title("Heat Flux vs Time")
xlabel("Time (min)")
ylabel("Voltage (V)")

% Plotting test 4
figure(4)
subplot(2, 1, 1)
plot(t4,Temperature4)
title("Internal Room Temperature vs Time")
xlabel("Time (min)")
ylabel("Temperature (deg C)")

subplot(2,1,2)
plot(t4, HFS4)
title("Heat Flux vs Time")
xlabel("Time (min)")
ylabel("Voltage (V)")

%% Plotting autocorrelation function
Numlags = 150;
figure(3)
autocorr(Temperature1, Numlags)
title("Sample Autocorrelation Function for Temperature (T1)")
mu_y1 = mean(Temperature1);

figure(4)
autocorr(HFS1, Numlags)
title("Sample Autocorrelation Function for HFS (T1)")
mu_x1 = mean(HFS1);

figure(5)
autocorr(Temperature2, Numlags)
title("Sample Autocorrelation Function for Temperature (T2)")
mu_y2 = mean(Temperature2);

figure(6)
autocorr(HFS2, Numlags)
title("Sample Autocorrelation Function for HFS (T2)")
mu_x2 = mean(HFS2);

figure(7)
autocorr(Temperature3, Numlags)
title("Sample Autocorrelation Function for Temperature (T3)")
mu_y3 = mean(Temperature3);

figure(8)
autocorr(HFS3, Numlags)
title("Sample Autocorrelation Function for HFS (T3)")
mu_x3 = mean(HFS3);

figure(9)
autocorr(Temperature4, Numlags)
title("Sample Autocorrelation Function for Temperature (T4)")
mu_y4 = mean(Temperature4);

figure(10)
autocorr(HFS4, Numlags)
title("Sample Autocorrelation Function for HFS (T4)")
mu_x4 = mean(HFS4);



%% Plotting Coherence Function different overlap
WINDOW = round(length(HFS4)/4);
NFFT=WINDOW;
OVERLAP1 = 0.9; OVERLAP2 = 0.8; OVERLAP3 = 0.7; OVERLAP4 = 0.6; %
fs = (1/60);
[cxy1,w1] = mscohere(HFS4, Temperature4,WINDOW,round(OVERLAP1*WINDOW),NFFT, fs);
[cxy2,w2] = mscohere(HFS4, Temperature4,WINDOW,round(OVERLAP2*WINDOW),NFFT, fs);
[cxy3,w3] = mscohere(HFS4, Temperature4,WINDOW,round(OVERLAP3*WINDOW),NFFT, fs);
[cxy4,w4] = mscohere(HFS4, Temperature4,WINDOW,round(OVERLAP4*WINDOW),NFFT, fs);

i=12; ii=16;

figure, hold on
plot(w1,cxy1,'k');
plot(w2,cxy2,'b');
plot(w3,cxy3,'r');
plot(w4,cxy4,'g');
set(gca,'fontsize',i)
title('Coherence Function for output (overlap effect)','Fontsize',ii)
ylabel('Coherence','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
legend('90 %','80 %','70 %','60 %','Location','SouthEast','Orientation','horizontal')

%% Plotting Coherence Function different windows
WINDOW1=50; WINDOW2=100; WINDOW3=150; WINDOW4=200; OVERLAP=0.6;
fs = (1/60);
[cxy1,w1] = mscohere(HFS4, Temperature4,WINDOW1,round(OVERLAP*WINDOW1),WINDOW1, fs);
[cxy2,w2] = mscohere(HFS4, Temperature4,WINDOW2,round(OVERLAP*WINDOW2),WINDOW2, fs);
[cxy3,w3] = mscohere(HFS4, Temperature4,WINDOW3,round(OVERLAP*WINDOW3),WINDOW3, fs);
[cxy4,w4] = mscohere(HFS4, Temperature4,WINDOW4,round(OVERLAP*WINDOW4),WINDOW4, fs);

i=12; ii=16;

figure
plot(w1,cxy1,'k'),hold on
plot(w2,cxy2,'b')
plot(w3,cxy3,'r')
plot(w4,cxy4,':g')
set(gca,'fontsize',i)
title('Coherence Function for output (window effect)','Fontsize',ii)
ylabel('Coherence','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
ylim([0 1])
legend('50', '100','150','200','Location','SouthEast','Orientation','horizontal')


%% Plotting the best one?
figure
mscohere(HFS4, Temperature4,150,round(0.7*150),150, fs);

%%
%pause % ,close


% for i = 1:length(window)
%     w = window(i);
%     nfft = w;
%     mscohere(HFS1, Temperature1, w, ol, nfft, colors(i))
%     title("Coherence Estimator via Welch" +  "," + " " +  "w = " + string(w))
%     hold on
% end

% figure(7)
% mscohere(HFS1, Temperature1, w, ol, nfft)
% title("Coherence Estaimte via Welch (T1)")
% 
% figure(8)
% mscohere(HFS2, Temperature2, w, ol, nfft)
% title("Coherence Estaimte via Welch (T2)")
% 
% figure(9)
% mscohere(HFS3, Temperature3, w, ol, nfft)
% title("Coherence Estaimte via Welch (T3)")
% 
% figure(10)
% mscohere(HFS4, Temperature4, w, ol, nfft)
% title("Coherence Estaimte via Welch (T4)")


%% Plotting FRF
w = 150;
nfft = 150;
ol = 50;

figure(9)
tfestimate(HFS1, Temperature1, w, ol, nfft)
title("Transfer Function Estimate via Welch (T1)")

figure(10)
tfestimate(HFS2, Temperature2, w, ol,nfft)
title("Transfer Function Estimate via Welch (T2)")

figure(11)
tfestimate(HFS3, Temperature3, w, ol,nfft)
title("Transfer Function Estimate via Welch (T3)")

figure(12)
tfestimate(HFS4, Temperature4, w, ol,nfft)
title("Transfer Function Estimate via Welch (T4)")

%% Plotting Spectogram
w = 200;
nfft = 200;
ol = 10;

figure(1)
spectrogram(HFS1, w, ol,nfft)
figure(2)
spectrogram(HFS2, w, ol,nfft)
figure(3)
spectrogram(HFS3, w, ol,nfft)
figure(4)
spectrogram(HFS4, w, ol,nfft)
figure(5)
spectrogram(Temperature1, w, ol,nfft)
figure(6)
spectrogram(Temperature2, w, ol,nfft)
figure(7)
spectrogram(Temperature3, w, ol,nfft)
figure(8)
spectrogram(Temperature4, w, ol,nfft)

%% BIC and RSS
load("SI_data_2_13.mat")
fs = (1/60);
X=x; Y=y;
N=size(y,1);
outputs=size(y,2);
output = 1;

models=[];

DATA=iddata(Y(:,output),X,1/fs);

minar = 1;
maxar = 50;


for order=minar:maxar
    %models{order}=armax(DATA,[order order+1 order 1 ]);
    models{order}=arx(DATA,[order order+1 0]);
end

Yp=cell(1,maxar); rss=zeros(1,maxar); BIC=zeros(1,maxar);

for order=minar:maxar
    BIC(order)=log(models{order}.noisevariance)+...
         (size(models{order}.parametervector,1)*log(N))/N;
end

for order=minar:maxar
    Yp{order}=predict(models{order},DATA,1);
    rss(order)=100*(norm(DATA.outputdata-Yp{order}.outputdata)^2)/(norm(DATA.outputdata)^2);
end

i=10; ii=14;


figure
subplot(2,1,1),plot(minar:maxar,BIC(minar:maxar),'-o')
xlim([minar maxar])
title('BIC criterion','Fontsize',ii)
ylabel('BIC','Fontsize',ii)
set(gca,'fontsize',i)
subplot(2,1,2),plot(minar:maxar,rss(minar:maxar),'-o')
xlim([minar maxar])
title('RSS/SSS criterion','Fontsize',ii)
ylabel('RSS/SSS (%)','Fontsize',ii)
xlabel('AR(n)','Fontsize',ii)
set(gca,'fontsize',i)

%% Frequency Stabilization Plot

[D,fn,z] = deal(zeros(maxar,maxar/2));

for order=minar:maxar
    clear num den
    num = models{order}.B;
    den = models{order}.A;
    [DELTA,Wn,ZETA,R,lambda]=disper_new(num,den,fs);
    qq = length(DELTA);
    D(order,1:qq) = DELTA';
    fn(order,1:qq) = Wn';
    z(order,1:qq) = ZETA';
end

%%
i=10; ii=14;

%s = 0.0005; % scaling factor
s = 10;

figure, hold on
for order=minar:maxar
    for jj=1:maxar/2
        %imagesc(fn(order,jj), order, z(order,jj));
        imagesc(s*fn(order,jj), order, z(order,jj))        
        %image(s*fn(order,jj), order, z(order,jj))
    end
end
%axis([0,5*fs/2,minar,maxar])
%axis([1,s*fs/2,minar,maxar])
colorbar,box on,
h = get(gca,'xtick');
set(gca,'xticklabel',h/s,'fontsize',i);
%set(gca,'xticklabel',h,'fontsize',i);
title('Frequency stabilization plot (colormap indicates damping ratio)','Fontsize',ii)
if isempty(x)
    ylabel('AR(n)','Fontsize',ii)
else
    ylabel('ARX(n,n)','Fontsize',ii)
end
xlabel('Frequency (Hz)','Fontsize',ii)
%xlim([0 fs/2])


%% Plotting Parametric FRF
[D,fn,z] = deal(zeros(maxar,maxar/2));
for order=minar:maxar
    clear num den
    num = models{order}.B;
    den = models{order}.A;
    [DELTA,Wn,ZETA,R,lambda]=disper_new(num,den,fs);
    qq = length(DELTA);
    D(order,1:qq) = DELTA';
    fn(order,1:qq) = Wn';
    z(order,1:qq) = ZETA';
end

order= 25; % select ARX/ARMA model orders 

disp('Natural Frequencies (Hz)');
disp(nonzeros(fn(order,:)))
 
disp('Damping Ratios (%)');
disp(nonzeros(z(order,:)))



df= 0.1;% delta w = fs/window

[MAG,PHASE,wp] = dbode(models{order}.B,models{order}.A,1/fs);  

i=10; ii=14;

figure
plot(wp/(2*pi),20*log10(abs(MAG)))
%plot(w,20*log10(abs(Txy)),'r')
xlim([0 fs/2])
set(gca,'fontsize',i)
feval('title',sprintf('Parametric FRF for selected orders - ARX(%d,%d)',order,order),...
        'Fontname','TimesNewRoman','fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)



%% Plotting Parametric vs Non-Parametric
WINDOW = 100;
NFFT = 100;
OVERLAP = 0.1;
[Txy,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs);

i=10; ii=14;

figure
plot(wp/(2*pi),20*log10(MAG)),hold on
plot(w,20*log10(abs(Txy)),'r')
xlim([0 fs/2])
set(gca,'fontsize',i)
title('Parametric (ARX based) vs non-parametric (Welch based) FRF comparison',...
'Fontname','TimesNewRoman','Fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
legend('Parametric','Welch based','Location','SouthEast','Orientation','vertical')

%% Confidence intervals
[Magh,phh,wp,sdmagh,sdphaseh]=bode(models{order});
wp=wp./(2*pi);

Magh=reshape(Magh,[size(Magh,3) 1]); 
sdmagh=reshape(sdmagh,[size(sdmagh,3) 1]); sdphaseh=reshape(sdphaseh,[size(sdphaseh,3) 1]);

%colA=[0 1 0];
%colB=[1 0 0];
colC=[0.8 0.8 0.8];

i=10; ii=14; 

a=2; % standard deviations

figure
set(gcf,'paperorientation','landscape','paperposition',[0.63 0.63 28.41 19.72]);
%subplot(3,2,1) 
box on, hold on
plot(wp,20*log10(Magh),'color',colC); 
plot(wp,20*log10(Magh-a*sdmagh),'color',colC);
plot(wp,20*log10(Magh+a*sdmagh),'color',colC);
%plot(w,20*log10(abs(Txy)),'--b')
%xlim([3 fs/2])
set(gca,'fontsize',i)
title('Parametric FRF with 95% confidence intervals','fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
%legend('Healhty','Damage A','Damage B','Damage C','Damage D','Damage E','Location','SouthEast',...
%    'Orientation','horizontal')
%legend([ph,pA,pB],'healthy','damage I','damage II','Location','SouthWest','Orientation','horizontal')
set(patch,'EdgeAlpha',0,'FaceAlpha',0)
patch([wp;flipud(wp)],[20*log10(abs(Magh-a*sdmagh));flipud(20*log10(Magh+a*sdmagh))],colC)

%% ACF Residuals
res=DATA.outputdata-Yp{order}.outputdata;

figure
acf_wn(res(order+1:end),100,0.8);
title('Residuals ACF','fontsize',12)
ylim([-0.5 0.5])

%% Predicting Signals
TT=0:1/fs:(length(y)-1)/fs;
figure
plot(TT,DATA.outputdata,'-o'), hold on
plot(TT,Yp{order}.outputdata,'*')
title('Model one-step-ahead prediction (*) vs actual signal (o)','fontsize',12)
ylabel('Signal','Fontsize',ii)
xlabel('Time (s)','Fontsize',ii)

%% Comparing models on diff data

% Creating arx model with data set 'a'
load("SI_data_2_27.mat")
fs = (1/60);
X=x; Y=y;
N=size(y,1);
outputs=size(y,2);
output = 1;

models=[];

DATA=iddata(Y(:,output),X,1/fs);

minar = 1;
maxar = 50;


for order=minar:maxar
    models{order}=arx(DATA,[order order+1 0]);
end

order= 25; % select ARX/ARMA model orders 

% Creating empty yp to store the prediction from arx model with new data 'b'
Yp=cell(1,maxar);

% Defining new data 'b'
load("SI_data_2_24.mat")
X=x; Y=y;
output = 1;
fs = (1/60);
DATA1=iddata(Y(:,output),X,1/fs);



% Prediction one step ahead for new data 'b'
Yp{order}=predict(models{order},DATA1,1);

i=10; ii=14; 
TT=0:1/fs:(length(y)-1)/fs;
figure
plot(TT,DATA1.outputdata,'-o'), hold on
plot(TT,Yp{order}.outputdata,'*')
title('Model one-step-ahead prediction (*) vs actual signal (o)','fontsize',12)
ylabel('Signal','Fontsize',ii)
xlabel('Time (s)','Fontsize',ii)


%% Comparing model frf with new data frf
[MAG,PHASE,wp] = dbode(models{order}.B,models{order}.A,1/fs);  

WINDOW = 100;
NFFT = 100;
OVERLAP = 0.1;
[Txy,w] = tfestimate(X,Y(:,output),WINDOW,round(OVERLAP*WINDOW),NFFT,fs);

i=10; ii=14;

figure
plot(wp/(2*pi),20*log10(MAG)),hold on
plot(w,20*log10(abs(Txy)),'r')
xlim([0 fs/2])
set(gca,'fontsize',i)
title('Parametric (ARX based) vs non-parametric (Welch based) FRF comparison',...
'Fontname','TimesNewRoman','Fontsize',ii)
ylabel('Magnitude (dB)','Fontsize',ii)
xlabel('Frequency (Hz)','Fontsize',ii)
legend('Parametric','Welch based','Location','SouthEast','Orientation','vertical')

%% residuals from prediction of one model to new data
res=DATA1.outputdata-Yp{order}.outputdata;

figure
acf_wn(res(order+1:end),100,0.8);
title('Residuals ACF','fontsize',12)
ylim([-0.5 0.5])











function [Delta,fn,z,R,lambda]=disper_new(num,den,Fs)

% num		: The numerator of the transfer function
% den		: The denominator of the transfer function
% Fs		: The sampling frequency (Hz)
% Delta	: The precentage dispersion
% fn		: The corresponding frequencies (Hz)
% z		: The corresponding damping (%)
% R		: The residues of the discrete system
% Mag		: The magnitude of the corresponding poles
% This function computes the dispersion of each frequency of a system. The System is  
% enetred as a transfer function. In case the order of numerator polynomial is greater than 
% that of the denominator the polynomial division is apllied, and the dispersion is considered at
% the remaine tf. The analysis is done using the Residuez routine of MATLAB.
% The results are printed in the screen in asceding order of frequencies.
% This routine displays only the dispersions from the natural frequencies (Complex Poles).

% REFERENCE[1]:  MIMO LMS-ARMAX IDENTIFICATION OF VIBRATING STRUCTURES - A Critical Assessment 
% REFERENCE[2]:  PANDIT WU

%--------------------------------------------
% Created	: 08 December 1999.
% Author(s)	: A. Florakis & K.A.Petsounis
% Updated	: 16 February 1999.
%--------------------------------------------

% Sampling Period
Ts=1/Fs;

% Calculate the residues of the Transfer Function
num=num(:).';
den=den(:).';

%---------------------------------------------------
% For Analysis with the contant term
%[UPOLOIPO,PILIKO]=deconv(fliplr(num),fliplr(den));
%UPOLOIPO=fliplr(UPOLOIPO);
%PILIKO=fliplr(PILIKO);
%---------------------------------------------------


[R,P,K]=residuez(num,den);
% keyboard
%OROS=PILIKO(1);
% Make rows columns
%R=R(:);P=P(:);K=K(:);
R=R(:);P=P(:);K=K(:);


% Distinction between Real & Image Residues  
[R,P,l_real,l_imag]=srtrp(R,P,'all');

% Construction of M(k) (Eq. 45 REF[1])
for k=1:length(P)
   ELEM=R./(ones(length(P),1)-P(k).*P);             % Construction of the terms Ri/1-pk*pi
   M(k)=R(k)*sum(ELEM);										 % Calculation of M(k)  
   clear ELEM
end

% Dispersion of Modes (Eq. 46 & 47 REF[1])
D_real=real(M(1:l_real));D_imag=M(l_real+1:l_imag+l_real);
D=[D_real';D_imag'+conj(D_imag)'];

% Precentage Dispersion (Eq. 48 REF[1])
%if ~isempty(K)
%   D=D(:).';
%   VARY=[K^2 2*K*OROS D]; 
%   Delta=100*VARY./sum(VARY);
	% tests   sum(Delta);Delta(1);Delta(2)
%   Delta=Delta(3:length(Delta))'
%else
%  disp('mhn mpeis')
	Delta=100*D./sum(D);
	%Delta=D_imag./sum(D_imag)
   sum(Delta);
   %dou=K^2/sum(D+K^2)
%end
%keyboard

% Sorting Dispersions by asceding Frequency 
lambda=P(l_real+1:l_imag+l_real);
Wn=Fs*abs(log(lambda));          % Corresponding Frequencies 
z= -cos(angle(log(lambda)));     % Damping Ratios
[Wn sr]=sort(Wn);
fn=Wn./(2*pi);                   % rad/sec==>Hz 
z=100*z(sr);

Delta=Delta(l_real+1:l_real+l_imag);
Delta=Delta(sr);

% Sorting Poles by asceding Frequency
lambda=lambda(sr);
R_imag_plus=R(l_real+1:l_real+l_imag);
R=R_imag_plus(sr);
%R=R.*Fs; 		% Residues for Impulse Invariance Method
%R=R./R(1);  	% Normalized Residues
   
Mag=abs(lambda);   % Magnitude of poles
Mag=Mag(sr);

%--------------------------------------------------------
% 				Results
%--------------------------------------------------------
form1= '%1d' ;
form2 = '%7.4e';  

if nargout==0,      
   % Print results on the screen. First generate corresponding strings:
   nmode = dprint([1:l_imag]','Mode',form1);
   wnstr = dprint(fn,'Frequency (Hz)',form2);
   zstr = dprint(z,'Damping (%)',form2);
   dstr = dprint(Delta,'Dispersion (%)',form2);
   rstr = dprint(R,'Norm. Residues ',form2);
   mrstr = dprint(lambda,'Poles',form1);
disp([nmode wnstr zstr dstr rstr mrstr	]);
else
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,P,nr,ni]=srtrp(R,P,flg)

% if flg='hf' ==> Real Residues & Imag Residues (from Real poles)
% else if flg='all' ==> Real Residues & Imag Residues (from positive poles) & Imag Residues (from negative Poles)

R_real=[];P_real=[];
R_imag_plus=[];P_imag_plus=[];
R_imag=[];P_imag=[];

for i=1:length(R)
   if imag(P(i))==0
      R_real=[R_real;R(i)];P_real=[P_real;P(i)];
   elseif imag(P(i))>0
      R_imag_plus=[R_imag_plus;R(i)];P_imag_plus=[P_imag_plus;P(i)];
   else
      R_imag=[R_imag;R(i)];P_imag=[P_imag;P(i)];
   end
end
switch flg
case 'all'
   R=[R_real;R_imag_plus;R_imag];P=[P_real;P_imag_plus;P_imag];
   nr=length(P_real);ni=length(P_imag);
case 'hf'
   P=[P_real;P_imag_plus];R=[R_real;R_imag_plus];
   nr=length(P_real);ni=length(P_imag);
end
end

function rk=acf_wn(x,maxlag,barsize)
% rk=acf_wn(x,maxlag,barsize);

R=xcorr(x,maxlag,'coeff');
rk=R(maxlag+2:end);
bar([1:maxlag],rk,barsize,'b'),hold
plot([1:maxlag],(1.96/sqrt(length(x))).*ones(maxlag,1),'r',[1:maxlag],(-1.96/sqrt(length(x))).*ones(maxlag,1),'r')
axis([0 maxlag+1 -1 1]),xlabel('Lag'),ylabel('A.C.F. ( \rho_\kappa )')
zoom on;hold
end




