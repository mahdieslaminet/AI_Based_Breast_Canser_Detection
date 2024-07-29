%% Start of Program
clc
clear
close all
warning off;

N=40;
max_it=100;
Vmax=0.5;
Vmin=-0.5;


%% Input Output Data
load IOdata

Data = IOdata;
X = Data(:,1:end-1);
Y = Data(:,end);
XN = X;
YN = Y;


%% Normalization

%
MinX = min(X); MaxX = max(max(abs(X)));
MinY = min(Y); MaxY = max(max(Y));
% %
XN=XN/MaxX;
YN=floor(YN/MaxY);

%% Test and Train Data
NumOfInputs = size(X,2);
NumOfOutputs = size(Y,2);
NumOfData = size(X,1);
NumOfHiddens=2*NumOfInputs+1;


TrPercent = 70;
DataNum = size(X,1);
TrNum = round(TrPercent * DataNum/100);
R = randperm(DataNum);
TrInx = R(1:TrNum);
TsInx = R(TrNum+1:end);

Xtr = XN(TrInx,:);
Ytr = YN(TrInx,:);

Xts = XN(TsInx,:);
Yts = YN(TsInx,:);


%% Network Training

[NetworkP ,BestCostP, BestChartP] = TrainUsingGA_Fcn(Xtr,Ytr,N,max_it,NumOfInputs,NumOfHiddens ,NumOfOutputs, Vmax, Vmin);

%% sim and mse


YtrNetP = sim(NetworkP,Xtr')';
YtsNetP = sim(NetworkP,Xts')';


MSEtrP = mse(Ytr-YtrNetP);
MSEtsP = mse(Yts-YtsNetP);


[c,cm,ind,per] = confusion(Yts',YtsNetP');
figure
plotconfusion(Yts',YtsNetP');

figure
plotroc(Yts',YtsNetP');

figure
plot(BestChartP,'r','LineWidth',2);
grid on;
title('Train MSE');