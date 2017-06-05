clear all;
close all;
clc;

% Using random seed to produce same results in each step
s = RandStream('mt19937ar','Seed',0);

load('data_iris.mat');

%no of Classes
classes = unique(Y);
num_features = size(X,2);

%Splitting Data
testsplits = 10;
numofClass = length(classes);

%initializing matrices
average_mu = zeros(numofClass, num_features);
avg_QDA_variances = zeros(numofClass, num_features);
avg_LDA_variances = zeros(1, num_features);
QDA_CCR = zeros(testsplits,1);
LDA_CCR = zeros(testsplits,1);
confusionmatrix_QDA = cell(testsplits,1);
confusionmatrix_LDA = cell(testsplits,1);

for i = 1:testsplits;
r_c = cvpartition(Y,'HoldOut',0.334);
train = training(r_c,1);
test = ~train;
X_train = X(find(train),:);
X_test = X(find(test),:);
Y_train = Y(train);
Y_test = Y(test);

%QDA Model
[QDAmodel]= kasimp93_QDA_train(X_train, Y_train, numofClass);
[Y_predict] = kasimp93_QDA_test(X_test, QDAmodel, numofClass);
CCR_QDA(i) = sum(Y_test == Y_predict)/50; 
C_QDA{i} = confusionmat(Y_predict,Y_test);

%LDA Model
[LDAmodel]= kasimp93_LDA_train(X_train, Y_train, numofClass);
[Y_predict] = kasimp93_LDA_test(X_test, LDAmodel, numofClass);
CCR_LDA(i) = sum(Y_test == Y_predict)/50;
C_LDA{i} = confusionmat(Y_predict,Y_test);


average_mu = average_mu + QDAmodel.Mu

for j = 1 : numofClass
        avg_QDA_variances(j,:) = avg_QDA_variances(j,:) + diag(QDAmodel.Sigma(:,:,j))';
end

    avg_LDA_variances = avg_LDA_variances + diag(LDAmodel.Sigmapooled)';
    
end
    
    %Mean of all 10 test CCRs
    mean_CCRofQDA = mean(CCR_QDA);
    mean_CCRofLDA = mean(CCR_LDA);
    
    %Standard Deviation of all 10 test CCRs
    std_CCRofQDA = std(CCR_QDA);
    std_CCRofLDA = std(CCR_LDA);
    
    % Mean vectors of training samples for each class (common to both LDA and QDA), averaged over 10 splits.
    average_mu = rdivide(average_mu,testsplits);
    
    % The variances of all the 4 dimensions (the diagonal terms of the covariance matrix) for each class of
    % training set (in QDA and LDA), averaged over 10 splits.
    avg_varofQDA = rdivide(avg_QDA_variances,testsplits); 
    avg_varofLDA = rdivide(avg_LDA_variances,testsplits);
     
    %confusion matrix of the best CCR(LDA)
    [~,idx] = max(CCR_LDA);
    conf_best_LDA = C_LDA{idx}
    
    %confusion matrix of the best CCR(LDA)
    [~,idx] = min(CCR_LDA);     
    conf_worst_LDA = C_LDA{idx}