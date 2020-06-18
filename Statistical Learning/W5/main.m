%% ECE 523
%Engineering Applications of Machine Learning and Data Analytics
%Homework 5 
%Professor: Gregory Ditzler
%Milad Panahi
%% P1
%% 10 Labeled%%
%% Generating the random data
clc
clear all
close all
format long
randgen([4 5],[9, 0; 0 9],[10 15],[5, 1.5; 1, 5.5],[15 10],[6, -1; -1, 4]);
clc
%% Initialization
load ALL_data
% the threshold is defined here
Threshold = 0.9;
%whole data
data1_all = x1';
data2_all = x2';
% Train Data
Train_Percenetage = 0.10;
Test_Percenetage = 0.2;
Number_Train_points = Train_Percenetage*(length(data1_all)+length(data2_all));
Number_Test_points = Test_Percenetage*(length(data1_all)+length(data2_all));
self_trainings = 5;
number_of_self_trainings_data = floor((1-Train_Percenetage-Test_Percenetage)*(length(data1_all)+length(data2_all))/self_trainings);
rng(1);
%% Pre-Processing and Discretization
data3_all = [data1_all;data2_all];
theclass_all = ones(size(data1_all,1)+size(data2_all,1),1);
theclass_all(1:size(data1_all,1)) = -1;
Data_Class_all = [data3_all,theclass_all];
Data_Class_all = Data_Class_all(randperm(size(Data_Class_all,1)),:);
Data_Class_Train = Data_Class_all([1:Number_Train_points],:);
Data_Class_Test  = Data_Class_all([end-Number_Test_points+1:end],:);

current_data =Number_Train_points;
for k = 1:self_trainings
%% Train an SVM classifier with Linear Kernel Function 
% Train an SVM classifier with Linear Kernel Function with training error
self_training_data_class = Data_Class_all([current_data+1:current_data+number_of_self_trainings_data],:);
current_data = current_data+number_of_self_trainings_data;
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
Loss_Trained_Linear  =Loss_Trained;
CompactSVMModel_2 = fitPosterior(CompactSVMModel,Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]));
[labels,PostProbs] = predict(CompactSVMModel_2,self_training_data_class(:,[1,2]));
errors = find(labels ~= self_training_data_class(:,[3]));
size_error(k) = length(errors)/number_of_self_trainings_data;
error_Test(k) = Loss_Trained_Linear;
for j=1:size(PostProbs)
     if (PostProbs(j,1)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1,2]),labels(j)]];
     elseif  (PostProbs(j,2)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1,2]),labels(j)]];
     else
     end
    
    
end

end
%%
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
error_Test(k+1) = Loss_Test;

%% cheecking with normal classifier without Semi-Supervised Learning
Data_Class_check  = Data_Class_all([1:end-Number_Test_points+1],:);
CompactSVMModel = fitcsvm(Data_Class_check(:,[1,2]),Data_Class_check(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
error_Test_just_check = Loss_Test;
error_Test;
plot([0:self_trainings],error_Test,'*r')
hold on
title('10% labeled')
plot([1],error_Test_just_check,'ob')
legend('Testing Error','Normal classifier without SSL')
xlabel('number of self trainings')
ylabel('testing error')
hold off
%% %25 labeled 
Train_Percenetage = 0.25;
Test_Percenetage = 0.15;
Number_Train_points = Train_Percenetage*(length(data1_all)+length(data2_all));
Number_Test_points = Test_Percenetage*(length(data1_all)+length(data2_all));
self_trainings = 5;
number_of_self_trainings_data = (1-Train_Percenetage-Test_Percenetage)*(length(data1_all)+length(data2_all))/self_trainings;
number_of_self_trainings_data = floor(number_of_self_trainings_data);
rng(1); % For reproducibility
%%

Data_Class_Train = Data_Class_all([1:Number_Train_points],:);
Data_Class_Test  = Data_Class_all([end-Number_Test_points+1:end],:);




%%    SVM Trainings

current_data =Number_Train_points;
for k = 1:self_trainings
%% Train an SVM classifier with Linear Kernel Function 
% Train an SVM classifier with Linear Kernel Function with training error
self_training_data_class = Data_Class_all([current_data+1:current_data+number_of_self_trainings_data],:);
current_data = current_data+number_of_self_trainings_data;
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
Loss_Trained_Linear  =Loss_Trained;
CompactSVMModel_2 = fitPosterior(CompactSVMModel,Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]));
[labels,PostProbs] = predict(CompactSVMModel_2,self_training_data_class(:,[1,2]));
errors = find(labels ~= self_training_data_class(:,[3]));
size_error(k) = length(errors)/number_of_self_trainings_data;
error_Test(k) = Loss_Trained_Linear;
for j=1:size(PostProbs)
     if (PostProbs(j,1)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1,2]),labels(j)]];
     elseif  (PostProbs(j,2)) >  Threshold
         Data_Class_Train = [Data_Class_Train;[self_training_data_class(j,[1,2]),labels(j)]];
     else
     end
    
    
end

end
%%
CompactSVMModel = fitcsvm(Data_Class_Train(:,[1,2]),Data_Class_Train(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
error_Test(k+1) = Loss_Test;

%% cheecking with normal classifier
Data_Class_check  = Data_Class_all([1:end-Number_Test_points+1],:);
CompactSVMModel = fitcsvm(Data_Class_check(:,[1,2]),Data_Class_check(:,[3]),'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Test = loss(CompactSVMModel,Data_Class_Test(:,[1,2]),Data_Class_Test(:,[3]));
error_Test_just_check = Loss_Test;
error_Test;
plot([0:self_trainings],error_Test,'*r')
hold on
plot([1],error_Test_just_check,'ob')
title('25% labeled')
legend('Testing Error','Normal classifier without SSL')
xlabel('number of self trainings')
ylabel('testing error')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%Conclusion Problem 1%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% It can be seen that, the SSL does not function well if we use a small number of labeled data.
% This happens because there is a high chance that we are not providing a good sample of data.
% However if we use more of traning data, the algorithm showed to
% perform better than a normal classifier without Semi-sepervised Learning


%% P2

%% Bank.csv
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('bank.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
%% making kfold
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of Final Kfold/training','Normal Classifier Without SSL','Final Kfold Error')
%% blood.csv
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('blood.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of a random Kfold/training','Normal Classifier Without SSL','Final Kfold Error')


%% breast-cancer.csv
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('breast-cancer.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of Final Kfold/training','Normal Classifier Without SSL','Final Kfold Error')


%% breast-cancer-wisc-diag
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('breast-cancer-wisc-diag.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of a random Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% breast-cancer-wisc-prog
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('breast-cancer-wisc-prog.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of Final Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% congressional-voting_train
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('congressional-voting_train.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of a random Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% conn-bench-sonar-mines-rocks
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('conn-bench-sonar-mines-rocks.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of Final Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% credit-approval.csv
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('credit-approval.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of a random Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% haberman-survival_test.csv
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('haberman-survival_test.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of Final Kfold/training','Normal Classifier Without SSL','Final Kfold Error')

%% heart-hungarian
clc
clear all
global iter
% data_S_Train = [1:20]';
data_S_Train = csvread('heart-hungarian.csv');
data_S_Train(find(data_S_Train(:,end)==0),end) = -1;
num_row = size(data_S_Train,1);
Number_kfold = 5;
num_row_fold = floor(num_row/Number_kfold);
iter = 1;
for q = 1:Number_kfold
 ind = 1:num_row;
 test = ind((q-1)*num_row_fold+1:(q)*num_row_fold);
 ind((q-1)*num_row_fold+1:(q)*num_row_fold) = [];
 data_S_Train_kfold = [data_S_Train(ind,:);data_S_Train(test,:)];
 [error] = crossfoldvalidation(data_S_Train_kfold);
 all_errors(q) = error;
 iter = iter+1;
end
error = sum(all_errors)/Number_kfold
hold on
plot(1,error,'b*')
legend('Testing Error of a random Kfold/training','Normal Classifier Without SSL','Final Kfold Error')
%%
% SSL's performance is highly dependent on the data distribution. However, in the case that we have a lack of labeled data, this
% method might be a good option, while a good percentage of labeled data needs be chosen for the initial data.
