
addpath(genpath('mil'))
addpath(genpath('prtools'))
addpath(genpath('minFunc_2012'))

%% load data for each subject and preprocessing
clear
clc
close all
prwarning(0)

% addpath data
addpath(genpath('../data'))

rng(3)
fold1=10;
fold2=5;


%% choose the data source
%Valence L=3
str='VDataOut3MIL%d.mat';
lw=3;
% %Valence L=5
% str='VDataOut5MIL%d.mat';
% lw=5;
% %Arousal L=3
% str='ADataOut3MIL%d.mat';
% lw=3;
% %Arousal L=5
% str='ADataOut5MIL%d.mat';
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
    
    totBagTrain=[cell2mat(BagTrainNegNorm); cell2mat(BagTrainPosNorm)];
    lab = genlab([sizerowN sizerowP]);
    idd=lw*ones(size(BagTrainNegNorm,1)+size(BagTrainPosNorm,1),1);
    bagid=genlab(idd);
    
    milDatasetTR=genmil(totBagTrain,lab,bagid);
    milDataset2TR = positive_class(milDatasetTR,2);
    
    %%%%%%% TESTTTT%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    
    C1 = bsxfun(@minus, testtot, mu1);
    sigma1(sigma1==0)=eps;
    TE1 = bsxfun(@rdivide, C1, sigma1);
    
    
    temp=(size(TE1,1)/lw);
    temp2=lw*ones(temp,1);
    
    BagTEnorm=mat2cell(TE1,temp2,sizecol);
    idpos=find(lab_test_out==2);
    idneg=find(lab_test_out==1);
    BagTENegNorm=[];
    BagTEPosNorm=[];
    for jj=1:numel(idneg)
        BagTENegNorm{jj,1}=BagTEnorm{idneg(jj),1};
    end
    for jj=1:numel(idpos)
        BagTEPosNorm{jj,1}=BagTEnorm{idpos(jj),1};
    end
    
    totBagTest=[cell2mat(BagTENegNorm); cell2mat(BagTEPosNorm)];
    lab = genlab([lw*numel(idneg) lw*numel(idpos)]);
    idd=lw*ones(size(BagTENegNorm,1)+size(BagTEPosNorm,1),1);
    bagid=genlab(idd);
    
    milDatasetTE=genmil(totBagTest,lab,bagid);
    milDataset2TE = positive_class(milDatasetTE,2);
    
    % Number of weak learners 100
    w=boosting_mil(milDataset2TR);
    yp=[];
    yp2=[];
    yp=milDataset2TE*w*labeld;
    
    for jj=1:numel(lab_test_out)
        if yp(jj)=='p'
            yp2(jj,1)=2;
        else
            yp2(jj,1)=1;
        end
    end
    
    
    accDDtest(out,1)=(sum(yp2==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,yp2);
    FDDmacro(out,1) = my_micro_macro( yp2 , lab_test_out);
    
    %save('Results_MILBoost_3MIL_Valence')
    
    
end

ConfDDtesttot=zeros(2,2);


for out=1:fold1
    ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
end


ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);

FDDmacro_tot=mean(FDDmacro);

save('Results_MILBoost_3MIL_Valence')
