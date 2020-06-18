function [error_final] = crossfoldvalidation(data_S_Train)
global iter
%% (Read Me) Fittng Different Kernel for SVM
%%%%%%%%%%%%%%%%%%%%%      Results are shown in RED     %%%%%%%%%%%%%%%%%%%%%
% four different Kernels are used
% 1) Gaussian Kernel with zero Training error ("BoxConstraint = Inf") \n\n
% 2) Gaussian Kernel with Training error (default Training error) \n\n
% 3) Linear Kernel  \n\n
% 4) Polynomial Kernel  \n\n
close all
format long
%% Generating the random data
%randgen([4 5],[9, 0; 0 9],[10 15],[5, 1.5; 1, 5.5],[15 10],[6, -1; -1, 4]);
clc
%% Initializing
%load ALL_data
% data_S_Train = csvread('bank.csv');
% 
Number_of_columns = size(data_S_Train,2);
% plot(data_S_Train(:,end),'*')
% the threshold is defined here
Threshold = 0.9;
%whole data
% Train Data
Train_Percenetage = 0.15;
Test_Percenetage = 0.20;

Number_Train_points = floor(Train_Percenetage*(size(data_S_Train,1)));
Number_Test_points = floor(Test_Percenetage*(size(data_S_Train,1)));
self_trainings = 5;
number_of_self_trainings_data = (1-Train_Percenetage-Test_Percenetage)*(size(data_S_Train,1))/self_trainings;
number_of_self_trainings_data = floor(number_of_self_trainings_data);

% data1_train = data1_all (1:end-100,:);
% data2_train = data2_all (1:end-100,:);
% Test Data
% data1_test = data1_all (end-100+1:end,:);
% data2_test = data2_all (end-100+1:end,:);
% Defining the classes
% data1 = data1_train;
% data2 = data2_train;
rng(1); % For reproducibility
% data3 = [data1;data2];
% data3_test = [data1_test;data2_test];
% theclass = ones(size(data1,1)+size(data2,1),1);
% theclass_test = ones(size(data1_test,1)+size(data2_test,1),1);
% theclass(1:size(data1,1)) = -1;
% theclass_test(1:size(data1_test,1)) = -1;
% Data_Class_Train = [data3,theclass];
% Data_Class_Train = Data_Class_Train(randperm(size(Data_Class_Train,1)),:)
% Data_Class_Test = [data3_test,theclass_test];

%%

Data_Class_all = data_S_Train;
Data_Class_all = Data_Class_all(randperm(size(Data_Class_all,1)),:);
Data_Class_Train = Data_Class_all([1:Number_Train_points],:);
Data_Class_Test  = Data_Class_all([end-Number_Test_points+1:end],:);




%%    SVM Trainings

current_data =Number_Train_points;
for k = 1:self_trainings
%% Train an SVM classifier with Linear Kernel Function 
% Train an SVM classifier with Linear Kernel Function with training error
self_training_data_class = Data_Class_all([current_data+1:current_data+number_of_self_trainings_data],:);
current_data = current_data+number_of_self_trainings_data;
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1:Number_of_columns-1]),Data_Class_Train(:,Number_of_columns),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,Data_Class_Test(:,[1:Number_of_columns-1]),Data_Class_Test(:,Number_of_columns));
Loss_Trained_Linear  =Loss_Trained;
CompactSVMModel_2 = fitPosterior(CompactSVMModel,Data_Class_Train(:,[1:Number_of_columns-1]),Data_Class_Train(:,Number_of_columns));
[labels,PostProbs] = predict(CompactSVMModel_2,self_training_data_class(:,[1:end-1]));
errors = find(labels ~= self_training_data_class(:,[3]));
size_error(k) = length(errors)/number_of_self_trainings_data;
error_Test(k) = Loss_Trained_Linear;
for j=1:size(PostProbs)
     if (PostProbs(j,1)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1:Number_of_columns-1]),labels(j)]];
     elseif  (PostProbs(j,2)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1:Number_of_columns-1]),labels(j)]];
     else
     end
    
    
end

end
%%
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1:Number_of_columns-1]),Data_Class_Train(:,Number_of_columns),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1:Number_of_columns-1]),Data_Class_Test(:,Number_of_columns));
error_Test(k+1) = Loss_Test;
error_final = error_Test(k+1);

%% cheecking with normal classifier
Data_Class_check  = Data_Class_all([1:end-Number_Test_points+1],:);
CompactSVMModel = fitcsvm(Data_Class_check(:,[1:Number_of_columns-1]),Data_Class_check(:,Number_of_columns),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1:Number_of_columns-1]),Data_Class_Test(:,Number_of_columns));
error_Test_just_check = Loss_Test;

error_Test;
if iter == 5
plot([0:self_trainings],error_Test,'*r')
hold on
plot([1],error_Test_just_check,'ob')
xlabel('number of self trainings')
ylabel('testing error')
end

%%
