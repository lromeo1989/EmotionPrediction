
function [mod1]=MI_Pattern_SVM(PBags,NBags,BC)

    neg_instances=cell2mat(NBags);
    pos_instances=cell2mat(PBags);
    
    lab_neg=ones(size(neg_instances,1),1);
    lab_pos=2*ones(size(pos_instances,1),1);
    
    XX=[neg_instances; pos_instances];
    YY=[lab_neg; lab_pos];
    select=1;
    count=0;
    nposBag=size(PBags,1);
    
    while(select~=0)
    count=count+1;

    mod1=fitcsvm(XX,YY,'KernelFunction','Linear','BoxConstraint',BC);
    for np=1:nposBag
        [out1, score1]=predict(mod1,PBags{np,1});
        y1{np,1}=out1;
        scorep{np,1}=score1(:,2);
    end
        
    for np=1:nposBag
       if sum(y1{np,1})==size(y1{np,1},1)
           [~,id_sel]=max(scorep{np,1});
           y1{np,1}(id_sel)=2;
       end
    end
    y1matpos=cell2mat(y1);
    Y1T=[lab_neg; y1matpos];
    
    if Y1T==YY; 
            select=0;
        else
           YY=Y1T;
           
    end  
        if count>11
            
            select=0;
%            disp('Exit')
            
        end    
    end
