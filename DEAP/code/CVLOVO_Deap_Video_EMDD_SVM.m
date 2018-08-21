addpath(genpath('DiverseDensity'))
addpath(genpath('EMDD'))
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
        feat_TR{ii,1}=sub_sel{1,ran11(ii)};
    end
    
    for jj=1:numel(ran22)
        feat_TE{jj,1}=sub_sel{1,ran22(jj)};
    end
    
    lab_TE=lab_sel(ran22);
    lab_TR=lab_sel(ran11);
    
    BagTrain=feat_TR;
    BagTest=feat_TE;
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

    npos=randperm(sizerowP);
    nposBag=size(BagTrainPos,1);
    DimBag=size(BagTrainPos{1,1},2);
    starting_bags=10;
    startingsTest=[];
    count=0;
    while(count~=starting_bags)
        curt=ceil(rand*nposBag);
        startingsTest=[startingsTest;BagTrainPosNorm{curt,1}];
        count=count+1;
    end

    tempsize=size(startingsTest);
    features_test=zeros(tempsize(1),DimBag);
    scales_test=zeros(tempsize(1),DimBag);
    dens_max_test=zeros(tempsize(1),1); 
    
    % you can use parfor here
    for iuu=1:tempsize(1)
        
   
        [h_t,s_t,mdens_t]=EMDD(startingsTest(iuu,:),DimBag,BagTrainPosNorm,BagTrainNegNorm);%,[4*DimBag,4*DimBag],[1e-5,1e-5,1e-7,1e-7],FCn);%,Epochs,Tol);
        features_test(iuu,:)=h_t;
        scales_test(iuu,:)=s_t;
        dens_max_test(iuu,:)=mdens_t;
        
    end

    [~, dm_id_test]=min(dens_max_test);
    
    features_vec_test=features_test(dm_id_test,:);
    scales_vec_test=scales_test(dm_id_test,:);
    
    
    %weight
    sk2=scales_vec_test.^2;
    

    %concept
    distNeg=[];
    distPos=[];
    max_conc=features_vec_test;
    for i1=1:length(BagTrainPosNorm)
        sz=size(BagTrainPosNorm{i1,1},1);
        diff2kp=(BagTrainPosNorm{i1,1}-repmat(max_conc,sz,1)).^2;
        distPos(i1,:)=min(sum(repmat(sk2,sz,1).*diff2kp,2));
    end
    
    for i2=1:length(BagTrainNegNorm)
        szz=size(BagTrainNegNorm{i2,1},1);
        diff2kn=(BagTrainNegNorm{i2,1}-repmat(max_conc,szz,1)).^2;
        distNeg(i2,:)=min(sum(repmat(sk2,szz,1).*diff2kn,2));
    end
        
        
    C1 = bsxfun(@minus, testtot, mu1);
    sigma1(sigma1==0)=eps;
    TE1 = bsxfun(@rdivide, C1, sigma1);
    
    
    DD_test=[];
    sz2=size(TE1,1);
    diff2kn=(TE1-repmat(max_conc,sz2,1)).^2;
    DD_test=min(sum(repmat(sk2,sz2,1).*diff2kn,2));   %% max
    if isnan(DD_test)
        disp ('Alert')
    end
    
    DD_train=[distNeg; distPos];
    lab_tr_tot=[ones(size(distNeg,1),1);2*ones(size(distPos,1),1)];

    
    Index3=crossvalind('Kfold',len3,fold3);


    SVM_testVal=[];
    lab_only_val_c=[];
    yptot=[];
    
    lab_teval_tot=[];
    
    % you can use parfor here
    for gin=1:fold3
        %disp(ue)
        DD_tr_val=[];
        DD_val=[];
        
        lab_tr_val=[];
        lab_val=[];
        
        iran11=find(Index3~=gin);
        iran22=find(Index3==gin);
        
        DD_tr_val=DD_train(iran11);
        DD_val=DD_train(iran22);
        
        lab_tr_val=lab_tr_tot(iran11);
        lab_val=lab_tr_tot(iran22);
        
        
        [TRtot2,mu2,sigma2]=zscore(DD_tr_val);
        
        C2 = bsxfun(@minus, DD_val, mu2);
        sigma2(sigma2==0)=eps;
        TE2 = bsxfun(@rdivide, C2, sigma2);
        
        ypp=[];
        
        lab_teval_tot=[lab_teval_tot;lab_val];
        
        for ins=1:numel(BC)
            
            
            SVMmodelin=fitcsvm(TRtot2,lab_tr_val,'KernelFunction','Linear','BoxConstraint',BC(ins));
            
            ypredval=predict(SVMmodelin,TE2);
            ypp=[ypp ypredval];
            
        end
        
        yptot=[yptot; ypp];
    end
    
    ysel=[];
    for ins=1:numel(BC)
        
        ysel=yptot(:,ins);
        fmacro(ins,1)  = my_micro_macro( ysel , lab_teval_tot);
    end
    
    [maxBC,locBC]=max(fmacro);
    
    %%test
    
    modSV=fitcsvm(DD_train,lab_tr_tot,'KernelFunction','Linear','BoxConstraint',BC(locBC));
    
    
    [ypred1, score11]=predict(modSV,DD_test);
    
    YY1(in1,1)=ypred1;
    score1tot(in1,1)=score11(:,2);
    labtesttotale(in1,1)=labtesttot_c;
    
end


accDDtest(out,1)=(sum(YY1==labtesttotale))/length(labtesttotale);
ConfDDtest{out,1}=confusionmat(labtesttotale,YY1);

FDDmacro(out,1)=my_micro_macro( YY1 , labtesttotale);

save('Results_EMDDSVM')

end



ConfDDtesttot=zeros(2,2);


for out=1:fold1
    ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
end


ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);


FDDmacro_mean=mean(FDDmacro);

save('Results_EMDDSVM')









