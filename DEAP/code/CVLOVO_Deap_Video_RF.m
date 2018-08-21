

rng(1)
% emo=1 valence emo=2 arousal
emo=1;
SVM_test_tot=[];
BC=[0.1 0.5 1 5 25 100];


for out=1:fold1
    
Index2=crossvalind('Kfold', len2, fold2);

sub_sel=[];
for jin=1:40
sub_sel{1,jin}=feat_tot{out,jin};
end

lab_sel=squeeze(labels(out,:,emo));

clc
disp('Test subjects')
disp(out)

for in1=1:fold2
    %disp('Test video');
    %disp(in1)
    
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
    testtot=cell2mat(BagTest);
    


    [TRtot,mu1,sigma1]=zscore(trainiam);
    sigma1(sigma1==0)=eps;
    C1 = bsxfun(@minus, testtot, mu1);
    TE1 = bsxfun(@rdivide, C1, sigma1);
    
    SVMmodel=TreeBagger(20,TRtot,lab_traintot_c,'NumPredictorsToSample','all','Method','classification');
  
   [ypred1]=predict(SVMmodel,TE1);
    ypred1=char(ypred1);
    ypred1=str2num(ypred1);
    

    YY1(in1,1)=ypred1;
    labtesttotale(in1,1)=labtesttot_c;
end

accDDtest(out,1)=(sum(YY1==labtesttotale))/length(labtesttotale);
ConfDDtest{out,1}=confusionmat(labtesttotale,YY1);
FDDmacro(out,1) = my_micro_macro(YY1, labtesttotale);
labtestout{out,1}=labtesttotale;

save('ResultsRF')

end    


labtestoutotale=cell2mat(labtestout);

ConfDDtesttot=zeros(2,2);


    for out=1:fold1     
        ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
    end



ConfDDtesttot=ConfDDtesttot/10;

ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);


FDDmacro_tot=mean(FDDmacro);


save('ResultsRF')














