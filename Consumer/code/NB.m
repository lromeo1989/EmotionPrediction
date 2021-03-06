
clear all
clc
close all

% addpath data
addpath(genpath('../data'))

rng(1)
fold1=10;
fold2=5;

%% choose the data source
%Valence
str='DataOut%d.mat';
%Arousal
%str='ADataOut%d.mat';

for out=1:10
    
    disp(out)
    
    matFileName = sprintf(str, out);
    
    load(matFileName)
    
    
    [TRtot,mu,sigma]=zscore(training_out);
    
    C = bsxfun(@minus, test_out, mu);
    sigma(sigma==0)=eps;
    TE = bsxfun(@rdivide, C, sigma);
    fmacro=[];
    
    
    model=fitcnb(TRtot,lab_training_out);
    
    [ypred1,scorepred1]=predict(model,TE);
    
    
    accDDtest(out,1)=(sum(ypred1==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,ypred1);
    
    FDDmacro(out,1) = my_micro_macro(ypred1, lab_test_out);
    ypredout{out,1}= ypred1;
    labtestout{out,1}=lab_test_out;
    
    save('Results_NB')
    
end


FDDmacrotot = mean(FDDmacro);

save('Results_NB')
