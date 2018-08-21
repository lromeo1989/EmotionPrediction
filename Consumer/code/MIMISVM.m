clear all
clc
close all

% addpath data
addpath(genpath('../data'))

rng(1)
fold1=10;
fold2=5;
BC=[0.1 0.5 1 5 25 100];
threshold=-2:.001:2;

%% choose the data source
%Valence L=3
str='DataOut3MIL%d.mat';
str2='DataIn3MIL%d.mat';
lw=3;
% %Valence L=5
% str='DataOut5MIL%d.mat';
% str2='DataIn5MIL%d.mat';
% lw=5;
%
% %Arousal L=3
% str='ADataOut3MIL%d.mat';
% str2='ADataIn3MIL%d.mat';
% lw=3;
% %Arousal L=5
% str='ADataOut5MIL%d.mat';
% str2='ADataIn5MIL%d.mat';
% lw=5;

for out=1:10
    
    disp(out)
    matFileName = sprintf(str, out);
    
    load(matFileName)
    
    PP=find(lab_training_out==2);
    NN=find(lab_training_out==1);
    
    BagTrainPos=[];
    BagTrainNeg=[];
    for iu=1:numel(PP)
        BagTrainPos{iu,1}=training_out{PP(iu),1};
    end
    
    for iu=1:numel(NN)
        BagTrainNeg{iu,1}=training_out{NN(iu),1};
    end
    BagTrainTot=[BagTrainNeg;BagTrainPos];
    traintot=cell2mat(BagTrainTot);
    [TRtot,mu1,sigma1]=zscore(traintot);
    
    seqp=[];
    seqn=[];
    testtot=cell2mat(test_out);
    
    for u1=1:size(BagTrainNeg,1)
        seqn=[seqn; size(BagTrainNeg{u1,1},1)];
    end
    for u2=1:size(BagTrainPos,1)
        seqp=[seqp; size(BagTrainPos{u2,1},1)];
    end
    sizecol=size(TRtot,2);
    sizerowP=size(cell2mat(BagTrainPos),1);
    sizerowN=size(cell2mat(BagTrainNeg),1);
    BagMatNeg=TRtot(1:sizerowN,:);
    BagMatPos=TRtot(sizerowN+1:end,:);
    BagTrainNegNorm=mat2cell(BagMatNeg, seqn, sizecol);
    BagTrainPosNorm=mat2cell(BagMatPos, seqp, sizecol);
    
    
    C1 = bsxfun(@minus, testtot, mu1);
    sigma1(sigma1==0)=eps;
    TE1 = bsxfun(@rdivide, C1, sigma1);
    
    temp=(size(TE1,1)/lw);
    temp2=lw*ones(temp,1);
    
    BagTEnorm=mat2cell(TE1,temp2,sizecol);
    
    %%%% Validation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fmacro=[];
    
    for in=1:5
        %disp(in)
        matFileName2 = sprintf(str2, in);
        
        load(matFileName2)
        
        PPVal=find(lab_training_in==2);
        NNVal=find(lab_training_in==1);
        
        BagTrainPosVal=[];
        BagTrainNegVal=[];
        for iu=1:numel(PPVal)
            BagTrainPosVal{iu,1}=training_in{PPVal(iu),1};
        end
        
        for iu=1:numel(NNVal)
            BagTrainNegVal{iu,1}=training_in{NNVal(iu),1};
        end
        BagTrainTotVal=[BagTrainNegVal;BagTrainPosVal];
        traintotVal=cell2mat(BagTrainTotVal);
        [TRtot2,mu2,sigma2]=zscore(traintotVal);
        
        seqpval=[];
        seqnval=[];
        testtotval=cell2mat(test_in);
        
        for u1=1:size(BagTrainNegVal,1)
            seqnval=[seqnval; size(BagTrainNegVal{u1,1},1)];
        end
        for u2=1:size(BagTrainPosVal,1)
            seqpval=[seqpval; size(BagTrainPosVal{u2,1},1)];
        end
        sizecolval=size(TRtot2,2);
        sizerowPval=size(cell2mat(BagTrainPosVal),1);
        sizerowNval=size(cell2mat(BagTrainNegVal),1);
        BagMatNegval=TRtot2(1:sizerowNval,:);
        BagMatPosval=TRtot2(sizerowNval+1:end,:);
        BagTrainNegNormVal=mat2cell(BagMatNegval, seqnval, sizecolval);
        BagTrainPosNormVal=mat2cell(BagMatPosval, seqpval, sizecolval);
        
        C2 = bsxfun(@minus, testtotval, mu2);
        sigma2(sigma2==0)=eps;
        TE2 = bsxfun(@rdivide, C2, sigma2);
        
        temp3=(size(TE2,1)/lw);
        temp4=lw*ones(temp3,1);
        
        BagTE2norm=mat2cell(TE2,temp4,sizecolval);
        
        max_score_MISVMneg=[];
        max_score_MISVMpos=[];
        SVM_testVal2=[];
        
        
        for idBC=1:numel(BC)
            
            MISVMmodelin=MI_Bag_SVM(BagTrainPosNormVal,BagTrainNegNormVal,BC(idBC));
            
            max_score_MISVM=[];
            for gin=1:size(BagTE2norm,1)
                [y,score_MISVM]=predict(MISVMmodelin,BagTE2norm{gin,1});
                max_score_MISVM(gin,1)=max(score_MISVM(:,2));
                
            end
            
            for ji=1:length(threshold)
                
                ypred2=ones(size(max_score_MISVM,1),1);
                ypred2(max_score_MISVM>threshold(ji))=2;
                fmacro(in,idBC,ji)= my_micro_macro( ypred2 , lab_test_in);
                
            end
            
            
        end
        
    end
    
    fmacroavg=squeeze(mean(fmacro));
    
    [v,l]=max(fmacroavg(:));
    [R, C]=ind2sub(size(fmacroavg),l);
    
    
    
    MISVMmodelout=MI_Bag_SVM(BagTrainPosNorm,BagTrainNegNorm,BC(R));
    
    SVM_test=[];
    
    for ginout=1:size(BagTEnorm,1)
        
        [~,score_MISVMtest]=predict(MISVMmodelout,BagTEnorm{ginout,1});
        SVM_test(ginout,1)=max(score_MISVMtest(:,2));
    end
    
    
    
    ypred1=ones(size(SVM_test,1),1);
    ypred1(SVM_test>threshold(C))=2;
    
    accDDtest(out,1)=(sum(ypred1==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,ypred1);
    FDDmacro(out,1) = my_micro_macro( ypred1 , lab_test_out);
    
    save('Results_MISVM')
    
end


ConfDDtesttot=zeros(2,2);


for out=1:fold1
    ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
end

ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);


FDDmacro_tot=mean(FDDmacro);

save('Results_MISVM')
