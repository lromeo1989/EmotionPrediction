
clear all
clc
close all

% addpath data
addpath(genpath('../data'))

rng(3)
fold1=10;
fold2=5;
BC=[20 50 100 200];

%% choose the data source
%Valence
str='VDataOut%d.mat';
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
    
     for in=1:5
        matFileName2 = sprintf('VDataIn%d%c%d.mat', in, '_',out);
        %matFileName2 = sprintf('DataIn%d.mat', in);
        
        load(matFileName2)
        
        [TRtot2,mu2,sigma2]=zscore(training_in);
        
        C2 = bsxfun(@minus, test_in, mu2);
        sigma2(sigma2==0)=eps;
        TE2 = bsxfun(@rdivide, C2, sigma2);
        
        ypp=[];
        yptot=[];
        
        for ins=1:numel(BC)
            
            modelVal=TreeBagger(BC(ins),TRtot2,lab_training_in,'NumPredictorsToSample','all','Method','classification');

            %SVMmodelin=fitcsvm(TRtot2,lab_training_in,'KernelFunction','Linear','BoxConstraint',BC(ins));
            
            ypredval=predict(modelVal,TE2);
             ypredval=char(ypredval);
            ypredval=str2num(ypredval);
            fmacro(in,ins)  = my_micro_macro(ypredval,lab_test_in);
        end
        
        
     end
    
    fmacroavg=mean(fmacro);
    [maxBC,locBC]=max(fmacroavg);
%     
%     hyp(out)=locBC;
    % number of weak learners: 20
    model=TreeBagger(BC(locBC),TRtot,lab_training_out,'NumPredictorsToSample','all','Method','classification');
    
    [ypred1,scoree]=predict(model,TE);
    ypred1=char(ypred1);
    ypred1=str2num(ypred1);
    scoretot_out{out,1}=scoree;
    
    accDDtest(out,1)=(sum(ypred1==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,ypred1);

    FDDmacro(out,1) = my_micro_macro(ypred1, lab_test_out);
    ypredout{out,1}= ypred1;
    labtestout{out,1}=lab_test_out;
    
    %save('Results_RF')

    
end

FDDmacrotot = mean(FDDmacro);
Acctot = mean(accDDtest);
ConfDDtesttot=zeros(2,2);


for out=1:10
    ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
end

save('Results_RF_valence_opt')

