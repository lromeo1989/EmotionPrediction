
addpath(genpath('mil'))
addpath(genpath('prtools'))
addpath(genpath('minFunc_2012'))

prwarning(0)
rng(1)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

threshold=-2:.001:2;
emo=2;
set=5;
SVM_test_tot=[];
BC=[0.1 0.5 1 5 25 100];  
for out=1:fold1
    
Index2=1:40;
sub_sel=[];
for jin=1:40
sub_sel{1,jin}=feat_tot_Bag{out,jin};
end


lab_sel=squeeze(labels(out,:,emo));

% clc
% disp('Test subjects')
% disp(out)

for in1=1:fold2
   % disp('Test video');
   % disp(in1)
    
    sub_test=[];
    sub_training=[];
    feat_offTE=[];
    feat_onTE=[];
    feat_offTR=[];
    feat_onTR=[];

    
    ran11=find(Index2~=in1);
    ran22=find(Index2==in1);
    
    feat_TR=[];
    feat_TE=[];
    
    for ii=1:numel(ran11)
        feat_TR{1,ii}=sub_sel{1,ran11(ii)};
    end
    
    for jj=1:numel(ran22)
        feat_TE{1,jj}=sub_sel{1,ran22(jj)};
    end
    
    lab_TE=lab_sel(ran22);
    lab_TR=lab_sel(ran11);
    
    BagTrain=reshape(feat_TR,[prod(size(feat_TR)) 1]);
    BagTest=reshape(feat_TE,[prod(size(feat_TE)) 1]);
    labtraintot_c=reshape(lab_TR,[prod(size(lab_TR)) 1]);
    lab_traintot_c=(labtraintot_c>5)+1;
    labtesttot_c=reshape(lab_TE,[prod(size(lab_TE)) 1]);
    labtesttot_c=(labtesttot_c>5)+1;
    
    trainiam=cell2mat(BagTrain);

    
    PP=find(lab_traintot_c==2);
    NN=find(lab_traintot_c==1);
    
    BagTrainPos=[];
    BagTrainNeg=[];
    for iu=1:numel(PP)
     BagTrainPos{iu,1}=BagTrain{PP(iu),1};
    end
    
    for iu=1:numel(NN)
     BagTrainNeg{iu,1}=BagTrain{NN(iu),1};
    end
    BagTrainTot=[BagTrainNeg;BagTrainPos];
    traintot=cell2mat(BagTrainTot);
    [TRtot,mu1,sigma1]=zscore(traintot);
    
    seqp=[];
    seqn=[];
    testtot=cell2mat(BagTest);

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
    idd=set*ones(39,1);
    bagid=genlab(idd);
   
    milDatasetTR=genmil(totBagTrain,lab,bagid);
    milDataset2TR = positive_class(milDatasetTR,2);
%%%%%%%%%%%%%%%%%%%% NO VALIDATION %%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%%%%%%%%%%


C1 = bsxfun(@minus, testtot, mu1);
sigma1(sigma1==0)=eps;
TE1 = bsxfun(@rdivide, C1, sigma1);
    
id1=(labtesttot_c==1);
id2=(labtesttot_c==2);

    lab = labtesttot_c.*ones(set,1);
    idd=set;
    bagid=genlab(idd);
    if labtesttot_c==1
        lab2={'negative'};
    else
       lab2={'positive'};
    end
    milDatasetTE=genmil(TE1,lab,bagid);
    milDataset2TE=setlablist(milDatasetTE,lab2);
%% MI-SVM

w=boosting_mil(milDataset2TR);
yp=milDataset2TE*w*labeld;

if yp=='positive' 
    yp2=2;
else
    yp2=1;
    
end

YY1(in1,1)=yp2;
labtesttotale(in1,1)=labtesttot_c;


end

accDDtest(out,1)=(sum(YY1==labtesttotale))/length(labtesttotale);
ConfDDtest{out,1}=confusionmat(labtesttotale,YY1);
FDDmacro(out,1) = my_micro_macro( YY1 , labtesttotale);
labtestout{out,1}=labtesttotale;

save('Results_milBoost')


end    

labtestoutotale=cell2mat(labtestout);

ConfDDtesttot=zeros(2,2);


    for out=1:fold1     
        ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
    end



ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);

FDDmacro_tot=mean(FDDmacro);

save('milBoostArousal5.mat')
    
save('Results_milBoost')




























