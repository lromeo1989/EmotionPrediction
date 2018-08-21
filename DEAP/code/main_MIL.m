
%% addpath
addpath(genpath('../data'))


clear
close all
clc

n=32;

%% load data 

load('Deap_Feat_Bag_Overlapped3_noHT.mat')
load('Deap_Feat_Bag_Overlapped5_noHT.mat')

clearvars -except feat_tot_Bag labels

for i=1:32
   for j=1:40
       temp=feat_tot_Bag{i,j}(:,1:2);
       temp(isnan(temp))=0;
       feat_tot_Bag{i,j}(:,1:2)=temp;
   end
end


fold1=32;
fold2=40;
len2=40;
len3=39;
fold3=39;


disp('CVLOVO Deap Video mi-SVM')
CVLOVO_Deap_Video_miPatternSVM

disp('CVLOVO Deap Video MI-SVM')
CVLOVO_Deap_Video_MISVM_Bag

disp('CVLOVO Deap Video EMDD-SVM')
CVLOVO_Deap_Video_EMDD_SVM

disp('CVLOVO Deap Video milBoost')
CVLOVO_Deap_Video_milBoost
