
%% addpath
addpath(genpath('../data'))


clear
close all
clc

n=32;



%% load data 
load('Deap_Feat_noHT.mat')

fold1=32;
fold2=40;
len2=40;
len3=39;
fold3=39;


%%
disp('CVLOVO Deap Video Naive Bayes')
CVLOVO_Deap_Video_NB

disp('CVLOVO Deap Video Baseline SVM')
CVLOVO_Deap_Video_BaselineSVM

disp('CVLOVO Deap Video Random Forrest')
CVLOVO_Deap_Video_RF
