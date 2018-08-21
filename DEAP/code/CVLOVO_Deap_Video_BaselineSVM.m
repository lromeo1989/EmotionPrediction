
rng(1)
% emo=1 valence emo=2 arousal
emo=1;
SVM_test_tot=[];
BC=[0.1 0.5 1 5 25 100];


for out=1:fold1
    
    Index2=crossvalind('Kfold', len2, fold2);
    
    sub_sel=[];
    for jin=1:40
        feat_tot{out,jin}(:,36)=[];
        sub_sel{1,jin}=feat_tot{out,jin};
    end
    
    lab_sel=squeeze(labels(out,:,emo));
    
    clc
    disp('Test subjects')
    disp(out)
    
    for in1=1:fold2
        %     disp('Test video');
        %     disp(in1)
        
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
        
        
        %Validation
        Index3=crossvalind('Kfold',len3,fold3);
        
        % disp('Starting validation stage')
        % tic
        
        
        Valfeat_onTR=[];
        Valfeat_offTR=[];
        Valfeat_onTE=[];
        Valfeat_offTE=[];
        
        SVM_testVal=[];
        lab_only_val_c=[];
        yptot=[];
        
        lab_teval_tot=[];
        
        % you can use parfor here
        for gin=1:fold3
            %disp(ue)
            trainVal=[];
            Val=[];
            
            iran11=find(Index3~=gin);
            iran22=find(Index3==gin);
            
            for ji=1:numel(iran11)
                trainVal{ji,1}=BagTrain{iran11(ji),1};
            end
            
            Val{ji,1}=BagTrain{iran22,1};
            
            trainVal2=trainVal;
            
            only_train=cell2mat(trainVal2);
            only_val=cell2mat(Val);
            
            lab_trval=lab_traintot_c(iran11,:);
            lab_teval=lab_traintot_c(iran22,:);
            
            [TRtot2,mu2,sigma2]=zscore(only_train);
            
            C2 = bsxfun(@minus, only_val, mu2);
            sigma2(sigma2==0)=eps;
            TE2 = bsxfun(@rdivide, C2, sigma2);
            
            ypp=[];
            
            lab_teval_tot=[lab_teval_tot;lab_teval];
            
            for ins=1:numel(BC)
                
                
                SVMmodelin=fitcsvm(TRtot2,lab_trval,'KernelFunction','Linear','BoxConstraint',BC(ins));
                
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
        
        
        SVMmodel=fitcsvm(TRtot,lab_traintot_c,'KernelFunction','Linear','BoxConstraint',BC(locBC));
        
        SVMmodel2=fitSVMPosterior(SVMmodel);
        SVMmodel=fitcnb(TRtot,lab_traintot_c);
        
        ypred1=predict(SVMmodel,TE1);
        
        [~,score11]=predict(SVMmodel,TE1);
        
        YY1(in1,1)=ypred1;
        score1tot(in1,1)=score11(:,2);
        labtesttotale(in1,1)=labtesttot_c;
    end
    
    accDDtest(out,1)=(sum(YY1==labtesttotale))/length(labtesttotale);
    ConfDDtest{out,1}=confusionmat(labtesttotale,YY1);
    FDDmacro(out,1) = my_micro_macro(YY1, labtesttotale);
    scorepredtot{out,1}=score1tot;
    labtestout{out,1}=labtesttotale;
    
    save('ResultsSVM')

end


score_tot=cell2mat(scorepredtot);
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


save('ResultsSVM')















