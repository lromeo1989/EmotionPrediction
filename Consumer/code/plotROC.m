clc; clear; close all;


% load('Results_RF_valence_opt')
% labtottest=cell2mat(labtestout);
% score_tot=cell2mat(scoretot_out);
% scoretot=score_tot(:,2);
% [X,Y]=perfcurve(labtottest,scoretot,2);
% figure
% plot(X,Y,'r')
% 
% clearvars -except X Y
% 
% 
% load('Results_miSVM_3MIL_Valence')
% labtottest=cell2mat(labtestout);
% scoretot=score_tot;
% [X,Y]=perfcurve(labtottest,scoretot,2);
% 
% hold on
% plot(X,Y,'g')
% 
% 
% 
% legend('RF','mi-SVM (3 windows)')
% xlabel('False positive rate')
% ylabel('True positive rate')



load('Results_RF_arousal_opt')
labtottest=cell2mat(labtestout);
score_tot=cell2mat(scoretot_out);
scoretot=score_tot(:,2);
[X,Y,T,AUC]=perfcurve(labtottest,scoretot,2);
figure
plot(X,Y,'r')

clearvars -except X Y


load('Results_miSVM_3MIL_Arousal')
labtottest=cell2mat(labtestout);
scoretot=score_tot;
[X,Y,T,AUC]=perfcurve(labtottest,scoretot,2);

hold on
plot(X,Y,'g')



legend('RF','mi-SVM (3 windows)')
xlabel('False positive rate')
ylabel('True positive rate')