function [Y_predict] = LDA_test(X_test, LDAmodel, numofClass)
%
% Testing for LDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% LDAmodel : the parameters of LDA classifier which has the follwoing fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your codes here:

load('data_iris.mat');

%Dimension of Data
D = size(X_test,1);
count = 0;
covariance = LDAmodel.Sigmapooled(:,:);
covariance_inv = inv(covariance);

for i= 1:numofClass
    %Finding Mean, Covariances and Prior Probability
     Mu = LDAmodel.Mu(i,:);
     pi = LDAmodel.Pi(i);
     h_map(:,i) = X_test*(Mu * covariance_inv)' - 0.5 * Mu * covariance_inv *Mu' + log(pi);
end

[posterior, Y_predict] = max(h_map,[],2);

end