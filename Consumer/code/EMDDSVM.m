addpath(genpath('DiverseDensity'))
addpath(genpath('EMDD'))

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
        %end
    end
    
    tempsize=size(startingsTest);
    features_test=zeros(tempsize(1),DimBag);
    scales_test=zeros(tempsize(1),DimBag);
    dens_max_test=zeros(tempsize(1),1);
    
    for iuu=1:tempsize(1)
        
        %% Finding Max Diverse Density...
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
    
    
    DD_test=[];
    for gin=1:size(BagTEnorm,1)
        sz2=size(BagTEnorm{gin,1},1);
        diff2kn=(BagTEnorm{gin,1}-repmat(max_conc,sz2,1)).^2;
        DD_test(gin,1)=min(sum(repmat(sk2,sz2,1).*diff2kn,2));   %% max
        if isnan(DD_test)
            disp ('Alert')
        end
    end
    
    DD_train=[distNeg; distPos];
    lab_tr_tot=[ones(size(distNeg,1),1);2*ones(size(distPos,1),1)];
    
    
    
    %%%% Validation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    fmacro=[];
    ypredval=[];
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
        
        distNegVal=[];
        distPosVal=[];
        max_conc=features_vec_test;
        for i1=1:length(BagTrainPosNormVal)
            sz=size(BagTrainPosNormVal{i1,1},1);
            diff2kpVal=(BagTrainPosNormVal{i1,1}-repmat(max_conc,sz,1)).^2;
            distPosVal(i1,:)=min(sum(repmat(sk2,sz,1).*diff2kpVal,2));
        end
        
        for i2=1:length(BagTrainNegNormVal)
            szz=size(BagTrainNegNormVal{i2,1},1);
            diff2knVal=(BagTrainNegNormVal{i2,1}-repmat(max_conc,szz,1)).^2;
            distNegVal(i2,:)=min(sum(repmat(sk2,szz,1).*diff2knVal,2));
        end
        
        DD_trainVal=[distNegVal; distPosVal];
        lab_tr_totVal=[ones(size(distNegVal,1),1);2*ones(size(distPosVal,1),1)];
        DD_testVal=[];
        for gin=1:size(BagTE2norm,1)
            sz2=size(BagTE2norm{gin,1},1);
            diff2knVal=(BagTE2norm{gin,1}-repmat(max_conc,sz2,1)).^2;
            DD_testVal(gin,1)=min(sum(repmat(sk2,sz2,1).*diff2knVal,2));   %% max
            if isnan(DD_testVal(gin,1))
                disp ('Alert')
            end
            
        end
        
        for idBC=1:numel(BC)
            
            
            SVMmodelin=fitcsvm(DD_trainVal,lab_tr_totVal,'KernelFunction','Linear','BoxConstraint',BC(idBC));
            
            ypredval=predict(SVMmodelin,DD_testVal);
            fmacro(in,idBC)= my_micro_macro( ypredval , lab_test_in);
            
        end
        
    end
    
    
    fmacroavg=(mean(fmacro));
    
    [R,C]=max(fmacroavg);
    
    
    SVMmodelout=fitcsvm(DD_train,lab_tr_tot,'KernelFunction','Linear','BoxConstraint',BC(C));
    SVMmodelout2=fitSVMPosterior(SVMmodelout);
    
    ypredtest=predict(SVMmodelout,DD_test);
    [~,score1tot]=predict(SVMmodelout2,DD_test);
    
    accDDtest(out,1)=(sum(ypredtest==lab_test_out))/length(lab_test_out);
    ConfDDtest{out,1}=confusionmat(lab_test_out,ypredtest);
    FDDmacro(out,1) = my_micro_macro( ypredtest , lab_test_out);
    scorepredtot{out,1}=score1tot;
    labtestout{out,1}=lab_test_out;
    
    save('Results_EMDDSVM')

end

ConfDDtesttot=zeros(2,2);


for out=1:fold1
    ConfDDtesttot=ConfDDtesttot+ConfDDtest{out,1};
end


ConfDDtesttot=(ConfDDtesttot./repmat(sum(ConfDDtesttot,2),1,2)*100);


accDDtest_mu=mean(accDDtest);
accDDtest_sigma=std(accDDtest);


FDDmacro_tot=mean(FDDmacro);

save('Results_EMDDSVM')


