clear all;
clc;

TrainsetVec = [1e1 1e2 1e3 1e4 1e5];
str = 12;
P = 16;
dd = 2;

trainDoF = 8;

TestSet = 1e4;

SNRmargin = 0;

trainSNR=10;
Kfactor = 3;

iterMax = 30;

subcarrier_badwidth=60*1e3;
subcarrier_spacing=15;
excess_delay=unifrnd(0,1e-6);


for indexN = 1:length(TrainsetVec)

    TrainSet = TrainsetVec(indexN);
    testDoF = trainDoF;

trainAns = [];
testAns  = [];

trainData1 = [];
testData1  = [];

trainData2 = [];
testData2  = [];

trainData3_Rician = [];
trainData3_Rayleigh = [];
testData3 = [];



parfor k=1:TrainSet

    whoRU = randi(2,1);
      IT = (randn(1,trainDoF) + 1i*randn(1,trainDoF));
      CR = exp(-1i*(pi*2)*(excess_delay)*(subcarrier_badwidth)*(subcarrier_spacing)*rand(1,trainDoF));
      xx = (IT * transpose(CR).^([0:P-1]));
    xx = xx / norm(xx);


    if whoRU == 1 % Rician

    ansVec = [1 0];

      IT_dominant = (randn(1,1) + 1i*randn(1,1)); 
      CR_dominant = exp(i*(pi*2)*rand(1,1));
      x_dominant = (IT_dominant * transpose(CR_dominant).^([0:P-1]));
      x_dominant = sqrt(10^(Kfactor/10)) * x_dominant; % K-factor 관련 정확한 수식 확인
      xx = xx + x_dominant;
      xx = xx / norm(xx);
      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];
      [Un, Sn, Vn] = makeHankel(xn);
      trainData2 = [trainData2 (diag(Sn))/norm(diag(Sn))];
      trainData3_Rician = [trainData3_Rician transpose(abs(ifft(xn)))];

      trainAns = [trainAns transpose(ansVec)];
    else  % Rayleigh

    ansVec = [0 1];

      xn = awgn(xx,trainSNR,'measured');

      trainData1 = [trainData1 transpose(abs((xn)))];
      [Un, Sn, Vn] = makeHankel(xn);
      trainData2 = [trainData2 (diag(Sn))/norm(diag(Sn))];
      trainData3_Rayleigh = [trainData3_Rayleigh transpose(abs(ifft(xn)))];

      trainAns = [trainAns transpose(ansVec)];
    end
end

skewness_Rician = mean(skewness(trainData3_Rician));
kurtosis_Rician = mean(kurtosis(trainData3_Rician));

skewness_Rayleigh = mean(skewness(trainData3_Rayleigh));
kurtosis_Rayleigh = mean(kurtosis(trainData3_Rayleigh));

skewness_Threshold = 0.5*(skewness_Rician+skewness_Rayleigh);
kurtosis_Threshold = 0.5*(kurtosis_Rician+kurtosis_Rayleigh);

skewness_Threshold;

kurtosis_Threshold;

net1 = fitnet(str*ones(1,dd));
net1.trainFcn = 'trainscg';
net1 = train(net1,(trainData1), (trainAns),'useGPU','yes');

net2 = fitnet(str*ones(1,dd));
net2.trainFcn = 'trainscg';
net2 = train(net2,(trainData2), (trainAns),'useGPU','yes');

parfor k=1:TestSet
   whoRU = randi(2,1);
     testSNR  = trainSNR + rand(1)*SNRmargin - SNRmargin/2;
      IT = (randn(1,testDoF) + 1i*randn(1,testDoF));
      CR = exp(-1i*(pi*2)*(excess_delay)*(subcarrier_badwidth)*(subcarrier_spacing)*rand(1,testDoF));
      xx = (IT * transpose(CR).^([0:P-1]));
    xx = xx / norm(xx);

    if whoRU == 1 % Rician

    ansVec = [1 0];

      IT_dominant = (randn(1,1) + 1i*randn(1,1)); 
      CR_dominant = exp(i*(pi*2)*rand(1,1));
      x_dominant = (IT_dominant * transpose(CR_dominant).^([0:P-1]));
      x_dominant = sqrt(10^(Kfactor/10)) * x_dominant; % K-factor 관련 정확한 수식 확인
      xx = xx + x_dominant;
      xx = xx / norm(xx);
      xn = awgn(xx,testSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];
      [Un, Sn, Vn] = makeHankel(xn);
      testData2 = [testData2 (diag(Sn))/norm(diag(Sn))];
      testAns = [testAns transpose(ansVec)];
      testData3 = [testData3 transpose(abs(ifft(xn)))];
    else  % Rayleigh

    ansVec = [0 1];

      xn = awgn(xx,testSNR,'measured');

      testData1 = [testData1 transpose(abs((xn)))];
      [Un, Sn, Vn] = makeHankel(xn);
      testData2 = [testData2 (diag(Sn))/norm(diag(Sn))];
      testAns = [testAns transpose(ansVec)];
      testData3 = [testData3 transpose(abs(ifft(xn)))];
    end
end

testResult1 = net1(testData1);
testResult1_mat = zeros(2,TestSet);

[ss ii] = max(testResult1);
for k=1:TestSet
testResult1_mat(ii(k),k)=1;
end

testResult2 = net2(testData2);
testResult2_mat = zeros(2,TestSet);

[ss ii] = max(testResult2);
for k=1:TestSet
testResult2_mat(ii(k),k)=1;
end


testResult3_mat = zeros(2,TestSet);
testResult4_mat = zeros(2,TestSet);

for k=1:TestSet
    if kurtosis_Threshold < kurtosis(testData3(:,k))
    testResult3_mat(1,k) = 1;
    else
    testResult3_mat(2,k) = 1;
    end

    if skewness_Threshold < skewness(testData3(:,k))
    testResult4_mat(1,k) = 1;
    else
    testResult4_mat(2,k) = 1;
    end
end


% testData3

detectionRate1(indexN) = sum(sum(testResult1_mat.*testAns)) / TestSet;
detectionRate2(indexN) = sum(sum(testResult2_mat.*testAns)) / TestSet;
detectionRate3(indexN) = sum(sum(testResult3_mat.*testAns)) / TestSet;
detectionRate4(indexN) = sum(sum(testResult4_mat.*testAns)) / TestSet;

end

figure(6)
semilogx(TrainsetVec, detectionRate2); hold on
semilogx(TrainsetVec, detectionRate1); hold on
semilogx(TrainsetVec, detectionRate4); hold on
semilogx(TrainsetVec, detectionRate3); hold on
xlabel('Number of training dataset')
ylabel('Detection rate')
legend('CR2Net','Deep learning-based (legacy)','Skewness-based','Kurtosis-based','location','best')
ylim([0.3 1])
xticks(TrainsetVec)
grid on
hold off
