C = 1;
B = 1;
%% imprt CSV files
close all;

data_S_Train = csvread('source_train.csv');
X_S_Train = data_S_Train(:,[1,2]);
Y_S_Train =  data_S_Train(:,end);

data_T_Train = csvread('target_train.csv');
X_T_Train = data_T_Train(:,[1,2]);
Y_T_Train=  data_T_Train(:,end);

data_T_Test = csvread('target_test.csv');
X_T_Test = data_T_Test(:,[1,2]);
Y_T_Test=  data_T_Test(:,end);

%% Finding W_S from SVM model using source training data

SVMModel = fitcsvm(X_S_Train,Y_S_Train,'KernelFunction', 'linear');
CVSVMModel = crossval(SVMModel);
classLoss = kfoldLoss(CVSVMModel);
Classifie_error_On_Source_Train  = classLoss;
W_S = SVMModel.Beta;


%% Finding W_T
% first we should find alpha(x) 
H  = (X_T_Train*X_T_Train').*(Y_T_Train*Y_T_Train');
for i = 1:size(H,2)
    f(i, 1) = (-1 + B * Y_T_Train(i) * dot(X_T_Train(i, :), W_S));
end
Aeq  = Y_T_Train';
beq  = 0;
A = [];
b = [];
lb = zeros(size(Aeq,2),1);
ub = C*ones(size(Aeq,2),1);
opts = optimoptions('quadprog','Display','iter');
x = quadprog(H,f,[],[],Aeq,beq,lb,ub,[],opts);
sum_sum = [0,0];
% now we could find W_T using alpha(x):
for i=1:size(x,1)
    sum_sum = sum_sum + x(i)*X_T_Train(i,:)*Y_T_Train(i);
end
w_new = B*W_S + sum_sum';

%% Lets look for target and testing accuracies:

%testing :
Y_predict = sign(X_T_Test*w_new);
a= find(Y_predict == Y_T_Test);
accuracy_test = length(a)/length(Y_predict);

%target:
Y_predict_train = sign(X_T_Train*w_new);
a_train= find(Y_predict_train == Y_T_Train);
accuracy_train = length(a_train)/length(Y_predict_train);



