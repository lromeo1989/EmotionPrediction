
function [mod1]=MI_Bag_SVM(PBags,NBags,BC)

neg_instances=cell2mat(NBags);
pos_instances=cell2mat(PBags);
nposBag=size(PBags,1);

for i=1:nposBag
    pos_Bag_avg(i,:)=mean(PBags{i,1},1);
end

lab_pos_Bag_avg=2*ones(size(pos_Bag_avg,1),1);
lab_neg_instances=ones(size(neg_instances,1),1);

X1=[neg_instances; pos_Bag_avg];
Y1=[lab_neg_instances; lab_pos_Bag_avg];
select=1;
count=0;
while(select~=0)
    count=count+1;
    
    mod1=fitcsvm(X1,Y1,'KernelFunction','Linear','BoxConstraint',BC);
    for np=1:nposBag
        [out1, score1]=predict(mod1,PBags{np,1});
        [mm id_sel]=max(score1(:,2));
        X1new(np,:)=PBags{np,1}(id_sel,:);
    end
    X1t=[];
    X1t=[neg_instances; X1new];
    
    if X1t==X1
        select=0;
    else
        X1=X1t;
    end
    
    if count>11
        
        select=0;
        disp('Exit')
        
    end
end