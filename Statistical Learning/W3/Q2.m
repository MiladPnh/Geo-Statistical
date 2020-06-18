%% (Read Me) Fittng Different Kernel for SVM
%%%%%%%%%%%%%%%%%%%%%      Results are shown in RED     %%%%%%%%%%%%%%%%%%%%%
% four different Kernels are used
% 1) Gaussian Kernel with zero Training error ("BoxConstraint = Inf") \n\n
% 2) Gaussian Kernel with Training error (default Training error) \n\n
% 3) Linear Kernel  \n\n
% 4) Polynomial Kernel  \n\n
clc
clear all

%% Generating the random data
randgen([4 5],[9, 0; 0 9],[10 15],[5, 1.5; 1, 5.5],[15 10],[6, -1; -1, 4]);
clc
%% Initializing
load ALL_data
%whole data
data1_all = x1';
data2_all = x2';
% Train Data
data1_train = data1_all (1:end-100,:);
data2_train = data2_all (1:end-100,:);
% Test Data
data1_test = data1_all (end-100+1:end,:);
data2_test = data2_all (end-100+1:end,:);
% Defining the classes
data1 = data1_train;
data2 = data2_train;
rng(1); % For reproducibility
data3 = [data1;data2];
data3_test = [data1_test;data2_test];
data3_all = [data1_all;data2_all];
theclass = ones(size(data1,1)+size(data2,1),1);
theclass_test = ones(size(data1_test,1)+size(data2_test,1),1);
theclass_all = ones(size(data1_all,1)+size(data2_all,1),1);
theclass(1:size(data1,1)) = -1;
theclass_test(1:size(data1_test,1)) = -1;
theclass_all(1:size(data1_all,1)) = -1;


%%    SVM Trainings


%% Train an SVM classifier with Gaussian Kernel Function zero trarining error

CompactSVMModel = fitcsvm(data3,theclass,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[-1,1]);
%Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(CompactSVMModel,xGrid);

%Plot the data and the decision boundary
figure
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(CompactSVMModel.IsSupportVector,1),data3(CompactSVMModel.IsSupportVector,2),'kd');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
Loss_Trained = loss(CompactSVMModel,data3,theclass);
Loss_Trained_No_training_error_Gaussian = Loss_Trained
str=sprintf('(Gaussian Kernel with zero training error) \n Trained Data Error = %d \n Support Vector Point  shown in Diamond  ',Loss_Trained);
title(str)
box on
savefig('Train_gaussian_Zero_Error.fig')
hold off

 figure
 hold on
 gscatter(data3_all(:,1),data3_all(:,2),theclass_all,'rb','.');
 [label,score] = predict(CompactSVMModel,data3_test);
 gscatter(data3_test(:,1),data3_test(:,2),label,'rb','oo');
 Test_Loss_with_No_training_error_Gaussian = loss(CompactSVMModel,data3_test,theclass_test)
 str=sprintf('(Gaussian Kernel with zero training error) \n Test data points with their decision \n are shown in circle , Test Error = %d',Test_Loss_with_No_training_error_Gaussian);
 title(str)
 box on
 savefig('Train_gaussian_Zero_Error.fig')
 print('Train_gaussian_Zero_Error','-dpdf','-fillpage')

 %% Train an SVM classifier with Gaussian Kernel Function with training error


CompactSVMModel = fitcsvm(data3,theclass,'KernelFunction','rbf',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,data3,theclass);
Loss_Trained_Gaussian_with_Training_Error = Loss_Trained
%Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(CompactSVMModel,xGrid);

%Plot the data and the decision boundary
figure
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
box on
ezpolar(@(x)1);
h(3) = plot(data3(CompactSVMModel.IsSupportVector,1),data3(CompactSVMModel.IsSupportVector,2),'kd');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal

str=sprintf(' (Gaussian Kernel with  training error) \n Trained Data Error = %d  \n Support Vector Point shown in Diamond  ',Loss_Trained);
title(str)
box on
savefig('Train_gaussian_with_Error.fig')
hold off

 figure
 hold on
 gscatter(data3_all(:,1),data3_all(:,2),theclass_all,'rb','.');
 [label,score] = predict(CompactSVMModel,data3_test);
 Loss_with_Test_error_Gaussian = loss(CompactSVMModel,data3_test,theclass_test)
 gscatter(data3_test(:,1),data3_test(:,2),label,'rb','oo');
 str=sprintf('(Gaussian Kernel with  training error) \n Test data points with their decision \n are shown in circle , Test Error = %d',Loss_with_Test_error_Gaussian);
title(str)
savefig('Train_gaussian_with_Error.fig')
print('Train_gaussian_with_Error','-dpdf','-fillpage')
%% Train an SVM classifier with Linear Kernel Function 
% Train an SVM classifier with Linear Kernel Function with training error

CompactSVMModel = fitcsvm(data3,theclass,'KernelFunction','linear',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,data3,theclass);
Loss_Trained_Linear  =Loss_Trained
%Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(CompactSVMModel,xGrid);

%Plot the data and the decision boundary
figure
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(CompactSVMModel.IsSupportVector,1),data3(CompactSVMModel.IsSupportVector,2),'kd');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
str=sprintf(' (Gaussian Kernel with  training error) \n Trained Data Error = %d  \n Support Vector Point shown in Diamond  ',Loss_Trained);
title(str)
box on
savefig('Train_Linear.fig')
hold off

figure
hold on
gscatter(data3_all(:,1),data3_all(:,2),theclass_all,'rb','.');
[label,score] = predict(CompactSVMModel,data3_test);
Loss_Linear  = loss(CompactSVMModel,data3_test,theclass_test);
Loss_Test_linear = Loss_Linear
gscatter(data3_test(:,1),data3_test(:,2),label,'rb','oo');
str=sprintf('(Gaussian Kernel with zero training error) \n Test data points with their decision \n are shown in circle , Test Error = %d',  Loss_Linear);
title(str)
box on
savefig('Train_linear.fig')
print('Train_linear','-dpdf','-fillpage')

%% Train an SVM classifier with Polynomial Kernel Function 

CompactSVMModel = fitcsvm(data3,theclass,'KernelFunction','polynomial',...
    'ClassNames',[-1,1]);
Loss_Trained = loss(CompactSVMModel,data3,theclass);
Loss_Trained_Polynomials = Loss_Trained
%Predict scores over the grid
d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(data3(:,1)):d:max(data3(:,1)),...
    min(data3(:,2)):d:max(data3(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
[~,scores] = predict(CompactSVMModel,xGrid);

%Plot the data and the decision boundary
figure
h(1:2) = gscatter(data3(:,1),data3(:,2),theclass,'rb','.');
hold on
ezpolar(@(x)1);
h(3) = plot(data3(CompactSVMModel.IsSupportVector,1),data3(CompactSVMModel.IsSupportVector,2),'kd');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legend(h,{'-1','+1','Support Vectors'});
axis equal
str=sprintf(' (Polynomial Kernel) \n Trained Data Error = %d  \n Support Vector Point shown in Diamond  ',Loss_Trained);
title(str)
box on
savefig('Train_Polynomial.fig')
hold off

figure
hold on
gscatter(data3_all(:,1),data3_all(:,2),theclass_all,'rb','.');
[label,score] = predict(CompactSVMModel,data3_test);
Loss_Polynommial = loss(CompactSVMModel,data3_test,theclass_test);
Loss_Polynommial_test  = Loss_Polynommial
%  table(data3_test(1:10),label(1:10),score(1:10,2),'VariableNames',...
%      {'TrueLabel','PredictedLabel','Score'})
gscatter(data3_test(:,1),data3_test(:,2),label,'rb','oo');
str=sprintf('(Polynomial Kernel) \n Test data points with their decision \n are shown in circle , Test Error = %d',  Loss_Polynommial);
title(str)
box on
savefig('Train_Polynomial.fig')
print('Train_Polynomial','-dpdf','-fillpage')

%% Making the Table
LastName = {'Gaussian Kernel with Zero Training Error';'Gaussian Kernel with Training Error';'Linear Kernel';'Polynomial Kernel'};
Training_Error = [Loss_Trained_No_training_error_Gaussian;Loss_Trained_Gaussian_with_Training_Error;Loss_Trained_Linear;Loss_Trained_Polynomials];
Testing_Error = [ Test_Loss_with_No_training_error_Gaussian;Loss_with_Test_error_Gaussian;Loss_Test_linear;Loss_Polynommial_test];
T = table(Training_Error,Testing_Error,...
    'RowNames',LastName)