
rng(1)
threshold=-2:.001:2;
% emo=1 valence emo=2 arousal
emo=1;
SVM_test_tot=[];
BC=[0.1 0.5 1 5 25 100];

for out=1:fold1
   
Index2=crossvalind('Kfold', len2, fold2);

sub_sel=[];
for jin=1:40
sub_sel{1,jin}=feat_tot_Bag{out,jin};
end

lab_sel=squeeze(labels(out,:,emo));

clc
disp('Test subjects')
disp(out)

for in1=1:fold2
    disp('Test video');
    disp(in1)
    
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
    


%%%%%%%%%%%%%%%%%%%% VALIDATION %%%%%%%%%%%%%%%%%%%%%%%%%


Index3=crossvalind('Kfold',len3,fold3);

% disp('Starting validation stage')
% tic


Valfeat_onTR=[];
Valfeat_offTR=[];
Valfeat_onTE=[];
Valfeat_offTE=[];

SVM_testVal=[];
lab_only_val_c=[];

% you can use parfor here
parfor ue=1:fold3
%disp(ue)


iran11=find(Index3~=ue);
iran22=find(Index3==ue);


ValTR_emo_onBag=[];
ValTR_emo_offBag=[];
ValTE_emo_onBag=[];
ValTE_emo_offBag=[];
PP2=PP;
NN2=NN;

if sum(iran22==PP)
iran22P=iran22;
iran22N=[];
PP2(iran22==PP2)=[];
lab_only_val_c=[lab_only_val_c; 2];
else
iran22N=iran22;
iran22P=[];
NN2(iran22==NN2)=[];
lab_only_val_c=[lab_only_val_c; 1];
end

iran11P=PP2;
iran11N=NN2;


%% SPLIT Bag-Instances 
for ii=1:size(iran11N,1) 
    ValTR_emo_offBag{ii,1}=BagTrain{iran11N(ii)};
end

for jj=1:size(iran11P,1) 
    ValTR_emo_onBag{jj,1}=BagTrain{iran11P(jj)};
end

for ii=1:size(iran22N,1) 
    ValTE_emo_offBag=BagTrain{iran22N(ii)};
end

for jj=1:size(iran22P,1) 
    ValTE_emo_onBag=BagTrain{iran22P(jj)};
end

BagValTrainNeg=ValTR_emo_offBag;
BagValTrainPos=ValTR_emo_onBag;
Vtrain_off=cell2mat(BagValTrainNeg);
Vtrain_on=cell2mat(BagValTrainPos);

only_train=[Vtrain_off; Vtrain_on];
lab_only_train=[ones(size(Vtrain_off,1),1);2*ones(size(Vtrain_on,1),1)];

BagValTestNeg=ValTE_emo_offBag;
BagValTestPos=ValTE_emo_onBag;

Vtest_off=BagValTestNeg; 
Vtest_on=BagValTestPos;

only_val=[Vtest_off; Vtest_on];
lab_only_val=[ones(size(Vtest_off,1),1);2*ones(size(Vtest_on,1),1)];

[TRtot2,mu2,sigma2]=zscore(only_train);

seqn=[];
seqp=[];


for u1=1:size(BagValTrainNeg,1)
   seqn=[seqn; size(BagValTrainNeg{u1,1},1)]; 
end  
for u2=1:size(BagValTrainPos,1)
   seqp=[seqp; size(BagValTrainPos{u2,1},1)]; 
end
sizecol=size(TRtot2,2);
sizerowP=size(cell2mat(BagValTrainPos),1);
sizerowN=size(cell2mat(BagValTrainNeg),1);
BagMatValNeg=TRtot2(1:sizerowN,:);
BagMatValPos=TRtot2(sizerowN+1:end,:);
BagTrainValNegNorm=mat2cell(BagMatValNeg, seqn, sizecol); 
BagTrainValPosNorm=mat2cell(BagMatValPos, seqp, sizecol); 

C2 = bsxfun(@minus, only_val, mu2);
sigma2(sigma2==0)=eps;
TE2 = bsxfun(@rdivide, C2, sigma2);

max_score_MISVMneg=[];
max_score_MISVMpos=[];
SVM_testVal2=[];
for idBC=1:numel(BC)

MISVMmodel=MI_Bag_SVM(BagTrainValPosNorm,BagTrainValNegNorm,BC(idBC));



[y,score_MISVM]=predict(MISVMmodel,TE2);
max_score_MISVM=max(score_MISVM(:,2));


SVM_testVal2=[SVM_testVal2 max_score_MISVM];  

end

SVM_testVal=[ SVM_testVal; SVM_testVal2];

end

FmacroVal=[];

SVM_test_val_sel=[];

for idBC=1:numel(BC)
SVM_test_val_sel=SVM_testVal(:,idBC);
fmacro=[];
for ji=1:length(threshold)    

    ypred2=ones(size(SVM_test_val_sel,1),1);
    ypred2(SVM_test_val_sel>threshold(ji))=2;
    fmacro  =[fmacro my_micro_macro(ypred2 , lab_only_val_c)];
    
end
FmacroVal=[FmacroVal; fmacro];


end


[v,l]=max(FmacroVal(:));
[R, C]=ind2sub(size(FmacroVal),l);



%%%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%%%%%%%%%%


C1 = bsxfun(@minus, testtot, mu1);
sigma1(sigma1==0)=eps;
TE1 = bsxfun(@rdivide, C1, sigma1);



%% MI-SVM
MISVMmodel_test=MI_Bag_SVM(BagTrainPosNorm,BagTrainNegNorm,BC(R));


[~,score_MISVMtest]=predict(MISVMmodel_test,TE1);
SVM_test=max(score_MISVMtest(:,2));


ypred1=ones(size(SVM_test,1),1);
ypred1(SVM_test>threshold(C))=2;
    
MISVMmodel_test2=fitSVMPosterior(MISVMmodel_test);
[~,score_MISVMtest2]=predict(MISVMmodel_test2,TE1);

YY1(in1,1)=ypred1;
score1tot(in1,1)=max(score_MISVMtest2(:,2));
labtesttotale(in1,1)=labtesttot_c;


end

accDDtest(out,1)=(sum(YY1==labtesttotale))/length(labtesttotale);
ConfDDtest{out,1}=confusionmat(labtesttotale,YY1);
FDDmacro(out,1) = my_micro_macro( YY1 , labtesttotale);
scorepredtot{out,1}=score1tot;
labtestout{out,1}=labtesttotale;

save('Results_MISVM')

end    



ConfDDtesttot=zeros(2,2);


    for out=1:fold1     
        ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
    end


ConfDDtesttot=ConfDDtesttot/10;

ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);

FDDmacro_tot=mean(FDDmacro);

save('Results_MISVM')
























