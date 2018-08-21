clear all
clc
close all

% addpath data
addpath(genpath('../data'))

rng(1)
fold1=10;
fold2=5;
BC=[0.1 0.5 1 5 25 100];

%% choose the data source
%Valence
str='DataOut%d.mat';
%Arousal
%str='ADataOut%d.mat';

for out=1:10
    
    
    matFileName = sprintf('DataOut%d.mat', out);
    
    load(matFileName)
    
    
    [TRtot,mu,sigma]=zscore(training_out);
    
    C = bsxfun(@minus, test_out, mu);
    sigma(sigma==0)=eps;
    TE = bsxfun(@rdivide, C, sigma);
    fmacro=[];
    
    for in=1:5
        
        matFileName2 = sprintf('DataIn%d.mat', in);
        
        load(matFileName2)
        
        [TRtot2,mu2,sigma2]=zscore(training_in);
        
        C2 = bsxfun(@minus, test_in, mu2);
        sigma2(sigma2==0)=eps;
        TE2 = bsxfun(@rdivide, C2, sigma2);
        
        ypp=[];
        yptot=[];
        
        for ins=1:numel(BC)
            
            
            SVMmodelin=fitcsvm(TRtot2,lab_training_in,'KernelFunction','Linear','BoxConstraint',BC(ins));
            
            ypredval=predict(SVMmodelin,TE2);
            fmacro(in,ins)  = my_micro_macro(ypredval,lab_test_in);
        end
        
        
    end
    
    fmacroavg=mean(fmacro);
    [maxBC,locBC]=max(fmacroavg);
    
    
    
    SVMmodel=fitcsvm(TRtot,lab_training_out,'KernelFunction','Linear','BoxConstraint',BC(locBC));
    SVModel2=fitSVMPosterior(SVMmodel);
    
    
    ypred1=predict(SVMmodel,TE);
    [~,scorepred1]=predict(SVModel2, TE);
    scorepredtot{out,1}=scorepred1;
    
    
    accDDtest(out,1)=(sum(ypred1==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,ypred1);
    FDDmacro(out,1) = my_micro_macro(ypred1, lab_test_out);
    labtestout{out,1}=lab_test_out;
    
    save('Results_SVM')
    
end

FDDmacrotot = mean(FDDmacro);

save('Results_SVM')


