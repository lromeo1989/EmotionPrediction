% % 
% addpath(genpath('/Users/Luca/Desktop/EmotionDetection/NewScript/lib'))
% %% load data for each subject and preprocessing
% clear all
% clc
% 
% %featExt
% % addpath(genpath('/Users/Luca/Desktop/EmotionDetection/Script'))
% % 
% % cd ('/Users/Luca/Desktop/EmotionDetection/Data')
% 
% load('29sogg.mat');
% load('dati.mat');
% 
% %%manca da tagliare e finestrare
% 
% %% Resampling spline
% 
% j=1:29;
% for j=j
%     nHRbase=1:1:length(HR(j).baseline);
%     NNTbase=1:1:length(BT(j).baseline);
% 
% end
% 
% j=1:87;
% 
% for j=j
%     disp('uu')
%     disp(j)
%     nHRhappy=1:1:length(hr(j).happy);
%     NNThappy=1:1:length(st(j).happy);
%     nGSRhappy=1:1:length(gsr(j).happy);
%     nRRhappy=1:1:length(rr(j).happy);
%     
%     nHRsad=1:1:length(hr(j).sadness);
%     NNTsad=1:1:length(st(j).sadness);
%     nGSRsad=1:1:length(gsr(j).sadness);
%     nRRsad=1:1:length(rr(j).sadness);
%     
%     
% 
%     
%     nHRhappysp=1:((length(hr(j).happy)-1)/(length(gsr(j).happy)-1)):length(hr(j).happy);
%     NNThappysp=1:((length(st(j).happy)-1)/(length(gsr(j).happy)-1)):length(st(j).happy);
%     nRRhappysp=1:((length(rr(j).happy)-1)/(length(gsr(j).happy)-1)):length(rr(j).happy);
%     
%     nHRsadsp=1:((length(hr(j).sadness)-1)/(length(gsr(j).sadness)-1)):length(hr(j).sadness);
%     NNTsadsp=1:((length(st(j).sadness)-1)/(length(gsr(j).sadness)-1)):length(st(j).sadness);
%     nRRsadsp=1:((length(rr(j).sadness)-1)/(length(gsr(j).sadness)-1)):length(rr(j).sadness);
%     
%     numGSRsad=nGSRsad;
%     numGSRhappy=nGSRhappy;
%     
%     close all
%     
%     hr(j).happy=zoh(nHRhappy,hr(j).happy,nHRhappysp);
%     hr(j).sadness=zoh(nHRsad,hr(j).sadness,nHRsadsp);
% 
%     st(j).happy=zoh(NNThappy,st(j).happy,NNThappysp);
%     st(j).sadness=zoh(NNTsad,st(j).sadness,NNTsadsp);
%    % plot(nRRhappy,rr(j).happy,'r')
%     rr(j).happy=zoh(nRRhappy,rr(j).happy,nRRhappysp);
%   %  hold on
%   %  plot(nRRhappysp,rr(j).happy)
%     rr(j).sadness=zoh(nRRsad,rr(j).sadness,nRRsadsp);
%     
% %     figure
% %     plot(GSR(j).happy,'r')
% %     hold on
%     outlierh=find(abs(diff(gsr(j).happy))>300);
%     numGSRhappy(outlierh+1)=[];
%     gsr(j).happy=interp1(numGSRhappy,gsr(j).happy(numGSRhappy),nGSRhappy,'spline');
% %     plot(GSR(j).happy)    
% 
%     
%       outliers=find(abs(diff(gsr(j).sadness))>300);
%     numGSRsad(outliers+1)=[];
%     gsr(j).sadness=interp1(numGSRsad,gsr(j).sadness(numGSRsad),nGSRsad,'spline');
%     
% %     figure
% %     plot(GSR(j).happy,'r')
% %     hold on;
% %     GSR(j).happy=smooth(GSR(j).happy,5)';
% %     plot(GSR(j).happy)
% %     
% %     GSR(j).happy=resample(GSR(j).happy,length(HR(j).happy),length(GSR(j).happy));
% %     GSR(j).sad=resample(GSR(j).sad,length(HR(j).sad),length(GSR(j).sad));
%     length(hr(j).happy);
%     length(rr(j).happy);
%     length(st(j).happy);
%     
%     length(hr(j).sadness);
%     length(rr(j).sadness);
%     length(st(j).sadness);
%     
%     gsr(j).sadness=smooth(gsr(j).sadness,5);
%     gsr(j).happy=smooth(gsr(j).happy,5);
%     
%     hr(j).sadness=smooth(hr(j).sadness,5);
%     hr(j).happy=smooth(hr(j).happy,5);
%     
%     rr(j).sadness=smooth(rr(j).sadness,5);
%     rr(j).happy=smooth(rr(j).happy,5);
%     
%     st(j).sadness=smooth(st(j).sadness,5);
%     st(j).happy=smooth(st(j).happy,5);
%     
%     data(j).HR_on=hr(j).happy;
%     data(j).RR_on=rr(j).happy;
%     data(j).GSR_on=gsr(j).happy;
%     data(j).BT_on=st(j).happy;
%     data(j).HR_off=hr(j).sadness;
%     data(j).RR_off=rr(j).sadness;
%     data(j).GSR_off=gsr(j).sadness;
%     data(j).BT_off=st(j).sadness;
% 
% end
%     i=0;
%     j=0;
%     while i<29
%     data(j+1).HR_baseline=HR(i+1).baseline;
%     data(j+2).HR_baseline=HR(i+1).baseline;
%     data(j+3).HR_baseline=HR(i+1).baseline;
%     data(j+1).BT_baseline=BT(i+1).baseline;
%     data(j+2).BT_baseline=BT(i+1).baseline;
%     data(j+3).BT_baseline=BT(i+1).baseline;
%     i=i+1;
%     j=j+3;
%     end
% 
% Nsub=1:87;
% %num=29;
% 
% % sub_on_features=zeros(num,Nfeat);
% % sub_baseline_features=zeros(num,Nfeat);
% 
% window=50;
% %%windowing % 30 sec %%20sec   
% 
% %windowing 5 sec --10samples
% for i = Nsub
%     
% %     A=[round(numel(data(i).RR_on(1,:))/54) round(numel(data(i).GSR_on(1,:))/200) round(numel(data(i).BT_on(1,:))/2)];
% %     B=[round(numel(data(i).RR_off(1,:))/54) round(numel(data(i).GSR_off(1,:))/200) round(numel(data(i).BT_off(1,:))/2)];
% %     len_on(i)=min(A);
% %     len_off(i)=min(B);
%     
%     data_wind(i).RR_on=vec2mat(data(i).RR_on(:,1),window);
%     data_wind(i).GSR_on=vec2mat(data(i).GSR_on(:,1),window);
%     data_wind(i).BT_on=vec2mat(data(i).BT_on(:,1),window);
%     data_wind(i).RR_off=vec2mat(data(i).RR_off(:,1),window);
%     data_wind(i).GSR_off=vec2mat(data(i).GSR_off(:,1),window);
%     data_wind(i).BT_off=vec2mat(data(i).BT_off(:,1),window);
%     
%     len_on(i)=size(data_wind(i).RR_on,1);
%     len_off(i)=size(data_wind(i).RR_off,1);
%     
% end    
% 
% 
% for i=1:87  
%    
%         
%     [featON, featBASE]=preprocessing_on(data(i),data_wind(i),len_on(i),window);
%     
%         
%     [featOFF]=preprocessing_off(data(i), data_wind(i),len_off(i),window);
%     
%     sub(i).on=featON;
%     sub(i).off=featOFF;
%     nani=sum(isnan(sub(i).on),2)>0;
%     nano=sum(isnan(sub(i).off),2)>0;
%  
%     sub(i).on(nani,:)=[];
%     sub(i).off(nani,:)=[]; 
%     
% end    
% 
% 
% ON=[sub(1).on(2:end-1,:); sub(2).on(2:end-1,:); sub(3).on(2:end-1,:); sub(4).on(2:end-1,:); sub(5).on(2:end-1,:);...
%      sub(6).on(2:end-1,:); sub(7).on(2:end-1,:); sub(8).on(2:end-1,:); sub(9).on(2:end-1,:); sub(10).on(2:end-1,:); ...
%      sub(11).on(2:end-1,:); sub(12).on(2:end-1,:); sub(13).on(2:end-1,:); sub(14).on(2:end-1,:); sub(15).on(2:end-1,:)]; %%happiness
%  
% OFF=[sub(1).off(2:end-1,:); sub(2).off(2:end-1,:); sub(3).off(2:end-1,:); sub(4).off(2:end-1,:); sub(5).off(2:end-1,:);...
%     sub(6).off(2:end-1,:); sub(7).off(2:end-1,:); sub(8).off(2:end-1,:); sub(9).off(2:end-1,:); sub(10).off(2:end-1,:); ...
%     sub(11).off(2:end-1,:); sub(12).off(2:end-1,:); sub(13).off(2:end-1,:); sub(14).off(2:end-1,:); sub(15).off(2:end-1,:)]; %%sadness
% 
% Tot=[ON;OFF];
% 
% response=[ones(size(ON,1),1);zeros(size(OFF,1),1)];
% Tot=[Tot response];
% 
% %indhappy=[1 25 34 37 40 49 55 72 79 5 11 14 26 32 38 41 50 56 68 71 3 27 30 33 45 51 54 60 66 69 10 22 25 ]
% 
% %% HAPPY
% classe_1=[sub(1).on(2:end-1,:);sub(25).on(2:end-1,:);sub(34).on(2:end-1,:);sub(37).on(2:end-1,:);sub(40).on(2:end-1,:);sub(49).on(2:end-1,:);...
%           sub(55).on(2:end-1,:);sub(72).on(2:end-1,:);sub(79).on(2:end-1,:);sub(5).on(2:end-1,:);sub(11).on(2:end-1,:);sub(14).on(2:end-1,:);...
%           sub(26).on(2:end-1,:);sub(32).on(2:end-1,:);sub(38).on(2:end-1,:);sub(41).on(2:end-1,:);sub(50).on(2:end-1,:);sub(56).on(2:end-1,:);...
%           sub(68).on(2:end-1,:);sub(71).on(2:end-1,:);sub(3).on(2:end-1,:);sub(27).on(2:end-1,:);sub(30).on(2:end-1,:);sub(33).on(2:end-1,:);...
%           sub(45).on(2:end-1,:);sub(51).on(2:end-1,:);sub(54).on(2:end-1,:);sub(60).on(2:end-1,:);sub(66).on(2:end-1,:);sub(69).on(2:end-1,:);...
%           sub(10).off(2:end-1,:);sub(22).off(2:end-1,:);sub(25).off(2:end-1,:);sub(40).off(2:end-1,:);sub(2).off(2:end-1,:);sub(5).off(2:end-1,:);...
%           sub(68).off(2:end-1,:);sub(27).off(2:end-1,:);sub(57).off(2:end-1,:)];% VALENCE>=6 e AROUSAL>=6 
% 
% %% SADNESS      
% classe_2=[sub(29).on(2:end-1,:);sub(47).on(2:end-1,:);sub(77).on(2:end-1,:);sub(72).on(2:end-1,:);sub(84).on(2:end-1,:);sub(34).off(2:end-1,:);...
%           sub(46).off(2:end-1,:);sub(49).off(2:end-1,:);sub(61).off(2:end-1,:);sub(70).off(2:end-1,:);sub(76).off(2:end-1,:);sub(82).off(2:end-1,:);...
%           sub(26).off(2:end-1,:);sub(69).off(2:end-1,:);sub(77).off(2:end-1,:);sub(9).off(2:end-1,:)];% VALENCE<=4 e AROUSAL<=4 
% 
% %% SAtisfaction      
% classe_3=[sub(7).on(2:end-1,:);sub(10).on(2:end-1,:);sub(19).on(2:end-1,:);sub(22).on(2:end-1,:);sub(28).on(2:end-1,:);sub(43).on(2:end-1,:);...
%           sub(46).on(2:end-1,:);sub(52).on(2:end-1,:);sub(76).on(2:end-1,:);sub(82).on(2:end-1,:);sub(85).on(2:end-1,:);sub(8).on(2:end-1,:);...
%           sub(53).on(2:end-1,:);sub(59).on(2:end-1,:);sub(65).on(2:end-1,:);sub(83).on(2:end-1,:);sub(86).on(2:end-1,:);sub(12).on(2:end-1,:);...
%           sub(42).on(2:end-1,:);sub(85).off(2:end-1,:);sub(86).off(2:end-1,:)];% VALENCE>=6 e AROUSAL<=4 
% 
% %% FEAR
% classe_4=[sub(63).on(2:end-1,:);sub(75).on(2:end-1,:);sub(1).off(2:end-1,:);sub(4).off(2:end-1,:);sub(13).off(2:end-1,:);sub(19).off(2:end-1,:);...
%           sub(31).off(2:end-1,:);sub(37).off(2:end-1,:);sub(43).off(2:end-1,:);sub(52).off(2:end-1,:);sub(58).off(2:end-1,:);sub(73).off(2:end-1,:);...
%           sub(11).off(2:end-1,:);sub(20).off(2:end-1,:);sub(29).off(2:end-1,:);sub(32).off(2:end-1,:);sub(41).off(2:end-1,:);sub(44).off(2:end-1,:);...
%           sub(47).off(2:end-1,:);sub(50).off(2:end-1,:);sub(53).off(2:end-1,:);sub(62).off(2:end-1,:);sub(65).off(2:end-1,:);sub(71).off(2:end-1,:);...
%           sub(12).off(2:end-1,:);sub(24).off(2:end-1,:);sub(30).off(2:end-1,:);sub(36).off(2:end-1,:);sub(39).off(2:end-1,:);sub(42).off(2:end-1,:);...
%           sub(45).off(2:end-1,:);sub(48).off(2:end-1,:);sub(51).off(2:end-1,:);sub(54).off(2:end-1,:);sub(63).off(2:end-1,:);sub(66).off(2:end-1,:);...
%           sub(69).off(2:end-1,:);sub(72).off(2:end-1,:)];% VALENCE<=4 e AROUSAL>=6 per i video sad
% 
%       
% %       
% Tot_classi=[classe_1;classe_2;classe_3;classe_4];
% Tot_classi=zscore(Tot_classi);
% response_classi=[zeros(size(classe_1,1),1);ones(size(classe_2,1),1);ones(size(classe_3,1),1)*2;ones(size(classe_4,1),1)*3];
% 
% clear all
% clc
% close all





load('ave29_each.mat')

% Valence
classe1=[classe_2;classe_3];
classe2=[classe_1;classe_4];


f = @kernelmi;
d_in=27;
d_max=34;

ConfNN=cell(d_max,10);
ConfSVM=cell(d_max,10);
ConfKNN=cell(d_max,10);
ConfDT=cell(d_max,10);
Mdl4=cell(d_max,10);

for j=1:d_max
  for i=1:10   
    ConfNN{j,i}=zeros(4,4);
    ConfSVM{j,i}=zeros(4,4);
    ConfKNN{j,i}=zeros(4,4);
    ConfDT{j,i}=zeros(4,4);
    Mdl4{j,i}=zeros(4,4);
%     ConfSVM{j,i}=zeros(cl,cl);
%     ConfKNN{j,i}=zeros(cl,cl);
  end 
end

for j=1:d_max
    ConfNNtot{j,1}=zeros(4,4);
    ConfSVMtot{j,1}=zeros(4,4);
    ConfKNNtot{j,1}=zeros(4,4);
    ConfDTtot{j,1}=zeros(4,4);
%     ConfSVM{j,i}=zeros(cl,cl);
%     ConfKNN{j,i}=zeros(cl,cl); 
end

%pool=parpool;
options = statset('UseParallel',1);
% 
% 
foldout=10;
foldin=5;
len1=size(classe1,1);
len2=size(classe2,1);


Indices1out = crossvalind('Kfold', len1, foldout);
Indices2out = crossvalind('Kfold', len2, foldout);


for out=1:foldout

disp(out)


c1val=classe1(Indices1out~=out,:);
c2val=classe2(Indices2out~=out,:);

totval=[c1val;c2val];
labval=[ones(size(c1val,1),1);2*ones(size(c2val,1),1)];

c1test=classe1(Indices1out==out,:);
c2test=classe2(Indices2out==out,:);


tottest=[c1test;c2test];
labttest=[ones(size(c1test,1),1);2*ones(size(c2test,1),1)];

nval1=size(c1val,1);
nval2=size(c2val,1);


Indices1 = crossvalind('Kfold', nval1, foldin);
Indices2 = crossvalind('Kfold', nval2, foldin);



% opts= struct;
% opts.depth= 9;
% opts.numTrees= 100;
% opts.numSplits= 5;
% opts.verbose= true;
% opts.classifierID= 2;
disp('Starting validation stage')
tic

for ue=1:foldin
disp(ue)

% Arousal=[classe_1;classe_4;classe_2;classe_3];
% lab_arousal=[zeros(size(classe_1,1),1);zeros(size(classe_4,1),1);ones(size(classe_2,1),1);ones(size(classe_3,1),1)];
% 
% 
% Valence=[classe_1;classe_3;classe_2;classe_4];
% lab_valence=[zeros(size(classe_1,1),1);zeros(size(classe_3,1),1);ones(size(classe_2,1),1);ones(size(classe_4,1),1)];

%for i=1:10


test1=c1val(Indices1==ue,:); 
train1=c1val(Indices1~=ue,:); 
test2=c2val(Indices2==ue,:); 
train2=c2val(Indices2~=ue,:); 
 

train_tot=[train1;train2];
test_tot=[test1;test2];

testiamo=[c1test;c2test];

lab_train=[ones(size(train1,1),1);2*ones(size(train2,1),1)];
lab_test=[ones(size(test1,1),1);2*ones(size(test2,1),1)];

lab_testiamo=[ones(size(c1test,1),1);2*ones(size(c2test,1),1)];

% lab_trainNN=[repmat([0 0 0 1],size(train1,1),1);repmat([0 0 1 0],size(train2,1),1);...
%     repmat([0 1 0 0],size(train3,1),1); repmat([1 0 0 0],size(train4,1),1);];
% lab_testNN=[repmat([0 0 0 1],size(test1,1),1);repmat([0 0 1 0],size(test2,1),1);...
%     repmat([0 1 0 0],size(test3,1),1); repmat([1 0 0 0],size(test4,1),1);];
z=[];
for j=1:size(train1,2)
    
    z=[z bsxfun(f,train_tot(:,j)',lab_train')];

end
[zval, ord]=sort(z,'descend');


% totran1=randperm(size(train_tot,1));
% totran2=randperm(size(test_tot,1));
% totran3=randperm(size(testiamo,1));


train_ran=train_tot(:,ord);
test_ran=test_tot(:,ord);
labtrain=lab_train(:,:);
labtest=lab_test(:,:);

testiamo_ran=testiamo(:,ord);
labtestiamo=lab_testiamo(:,:);

parfor num=d_in:d_max
    %disp(num)
    
    
    TR=train_ran(:,1:num);
    [TR,mu,sigma]=zscore(TR);
    TE=test_ran(:,1:num);
    C = bsxfun(@minus, TE, mu);
    TE = bsxfun(@rdivide, C, sigma);
    %TEE=testiamo_ran(:,1:num);
    % NN
%     Mdl = fitcNN(TR,labtrain);
%     yout=predict(Mdl,TE);
%     accNN(ue,num)=(sum(yout==labtest))/length(labtest);
%     ConfNN{num,ue}=confusionmat(labtest,yout);
%     net = cascadeforwardnet(10);
%     net = train(net,TR.',labtrainNN.');
%     %view(net)
%     yout = net(TE.');
%     yind = vec2ind(yout);
%     accNN(ue,num) = sum(labtest == yind.')/numel(labtest);
%     ConfNN{num,ue}=confusionmat(labtest,yind.');

    %SVM Linear
    %t = templateSVM('KernelFunction','linear');
%     A=fitcecoc(TR,labtrain,'Options',options,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','ShowPlots',false,'Verbose',0));
%     Mdl1{ue,num}=A;
%     yout2 = predict(A,TE);
%     accSVM(ue,num)=(sum(yout2==labtest))/length(labtest);
%     ConfSVM{ue,num}=confusionmat(labtest,yout2);

    %SVM Gaussian
    B=fitcsvm(TR, labtrain, 'KernelFunction','Gaussian','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0));
    %B=fitcecoc(TR,labtrain,'Learners',t,'Options',options);%,'OptimizeHyperparameters',paramsSVM,'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
    Mdl2{ue,num}=B;
    yout2 = predict(B,TE);
    accSVMG(ue,num)=(sum(yout2==labtest))/length(labtest);
    ConfSVMG{ue,num}=confusionmat(labtest,yout2);

    
    %KNN
    params = hyperparameters('fitcknn',TR,labtrain);
    params(3:5)=[];
    params(1,1).Range(1,2)=20;
    params(2,1).Range=params(2,1).Range(1,1:6);
    C=fitcknn(TR,labtrain);%,'OptimizeHyperparameters',params,'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
    Mdl3{ue,num}=C;
    yout3 = predict(C,TE);
    accKNN(ue,num)=(sum(yout3==labtest))/length(labtest);
    ConfKNN{ue,num}=confusionmat(labtest,yout3);

    
    %DecisionForest
%     Mdl4= forestTrain(TR, labtrain, opts);
%     yout4 = forestTest(Mdl4, TE);
%     accDT(ue,num)=(sum(yout4==labtest))/length(labtest);
%     ConfDT{num,ue}=confusionmat(labtest,yout4);

    %Decision Tree
     
     D= fitctree(TR, labtrain);%, 'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
     Mdl4{ue,num}=D;
     yout4 = predict(D, TE);
     accDT(ue,num)=(sum(yout4==labtest))/length(labtest);
     ConfDT{ue,num}=confusionmat(labtest,yout4);
     %yout41=predict(A,TEE); 
     
end

end

toc
disp('Completed Validation Stage')
disp(toc)
timeVal(out)=toc;

ACCKNN=mean(accKNN,1);
%ACCSVM=mean(accSVM,1);
ACCSVMG=mean(accSVMG,1);
ACCDT=mean(accDT,1);

[KNNmax, ordKNN]=max(ACCKNN);
[DTmax, ordDT]=max(ACCDT);
%[SVMmax, ordSVM]=max(ACCSVM);
[SVMGmax, ordSVMG]=max(ACCSVMG);
% ordDT=34;
% ordKNN=34;
% ordSVM=34;

disp('Final Model Testing')
tic

for j=1:size(totval,2)
    
    z2(j)=bsxfun(f,totval(:,j)',labval');

end
[zval2, ord2]=sort(z2,'descend');


%% TEST DT
totvalord=totval(:,ord2(1:ordDT));
tottestord=tottest(:,ord2(1:ordDT));
[TRtot,mu1,sigma1]=zscore(totvalord);
temp1 = bsxfun(@minus, tottestord, mu1);
tottestord = bsxfun(@rdivide, temp1, sigma1);

AA= fitctree(TRtot, labval);%,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
Mdl1test{out,1}=AA;
yout41 = predict(AA, tottestord);
accDTtest{out,1}=(sum(yout41==labttest))/length(labttest);

%% TEST K-NN

totvalord2=totval(:,ord2(1:ordKNN));
tottestord2=tottest(:,ord2(1:ordKNN));
[TRtot2,mu2,sigma2]=zscore(totvalord2);
temp2 = bsxfun(@minus, tottestord2, mu2);
tottestord2 = bsxfun(@rdivide, temp2, sigma2);

params2 = hyperparameters('fitcknn',TRtot2,labval);
params2(3:5)=[];
params2(1,1).Range(1,2)=20;
params2(2,1).Range=params2(2,1).Range(1,1:6);
AA2= fitcknn(TRtot2, labval);%,'OptimizeHyperparameters',params2,'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0));
Mdl2test{out,1}=AA2;
yout42 = predict(AA2, tottestord2);
accKNNtest{out,1}=(sum(yout42==labttest))/length(labttest);

%% TEST Linear Kernel SVM

% totvalord3=totval(:,ord2(1:ordSVM));
% tottestord3=tottest(:,ord2(1:ordSVM));
% tt = templateSVM('KernelFunction','linear');
% [TRtot3,mu3,sigma3]=zscore(totvalord3);
% temp3 = bsxfun(@minus, tottestord3, mu3);
% tottestord3 = bsxfun(@rdivide, temp3, sigma3);
% 
% 
% AA3=fitcecoc(TRtot3,labval,'Learners',tt,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('Optimizer','randomsearch','ShowPlots',false,'Verbose',0));
% Mdl3test{out,1}=AA3;
% yout43 = predict(AA3,tottestord3);
% accSVMtest{out,1}=(sum(yout43==labttest))/length(labttest);

%% TEST Gaussian Kernel SVM

totvalord4=totval(:,ord2(1:ordSVMG));
tottestord4=tottest(:,ord2(1:ordSVMG));
tt = templateSVM('KernelFunction','gaussian');
[TRtot4,mu4,sigma4]=zscore(totvalord4);
temp4 = bsxfun(@minus, tottestord4, mu4);
tottestord4 = bsxfun(@rdivide, temp4, sigma4);
AA4=fitcsvm(TRtot4, labval, 'KernelFunction','Gaussian','OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus','ShowPlots',false,'Verbose',0));
Mdl4test{out,1}=AA4;
yout44 = predict(AA4,tottestord4);
accSVMGtest{out,1}=(sum(yout44==labttest))/length(labttest);


ConfDTtest{out,1}=confusionmat(labttest,yout41);
ConfKNNtest{out,1}=confusionmat(labttest,yout42);
% ConfSVMtest{out,1}=confusionmat(labttest,yout43);
ConfSVMGtest{out,1}=confusionmat(labttest,yout44);

disp('Completed Final Model Testing')
toc
disp(toc)
timeTe(out)=toc;
save('HypOPBayesian_FS.mat')

end



ConfDTtesttot=zeros(2,2);
ConfKNNtesttot=zeros(2,2);
ConfSVMGtesttot=zeros(2,2);


    for out=1:10
        
        ConfDTtesttot=ConfDTtesttot+ConfDTtest{out,1};
        ConfKNNtesttot=ConfKNNtesttot+ConfKNNtest{out,1};
        %ConfNNtot{i,1}=ConfNNtot{i,1}+ConfNN{i,j};
        ConfSVMGtesttot=ConfSVMGtesttot+ConfSVMGtest{out,1};
    end

% Tot_classi=[Tot_classi response_classi];
% 
% classificationLearner

ConfDTtesttot=ConfDTtesttot/10;
ConfKNNtesttot=ConfKNNtesttot/10;
ConfSVMGtesttot=ConfSVMGtesttot/10;

ConfDTtesttot=(ConfDTtesttot./repmat(sum(ConfDTtesttot,2),1,2)*100);
ConfKNNtesttot=(ConfKNNtesttot./repmat(sum(ConfKNNtesttot,2),1,2)*100);
ConfSVMGtesttot=(ConfSVMGtesttot./repmat(sum(ConfSVMGtesttot,2),1,2)*100);


accDTtest_mu=mean(cell2mat(accDTtest));
accDTtest_sigma=std(cell2mat(accDTtest));

accKNNtest_mu=mean(cell2mat(accKNNtest));
accKNNtest_sigma=std(cell2mat(accKNNtest));

accSVMGtest_mu=mean(cell2mat(accSVMGtest));
accSVMGtest_sigma=std(cell2mat(accSVMGtest));


delete(pool)



%view(Mdl4{10,1},'Mode','graph');     

