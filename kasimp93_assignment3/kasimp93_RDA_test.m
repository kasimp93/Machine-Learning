function [Y_predict] = kasimp93_RDA_test(X_test, RDAmodel, numofClass)
%
% Testing for RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix 
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:

%Dimension of Data
D = size(X_test,1);
count = 0;

%Inverting the covariance matrix of LDA for RDA
covariance = RDAmodel.Sigmapooled(:,:);
covariance_inv = inv(covariance);

for i= 1:numofClass
    
    %Finding Mean and posterior Probability
     Mu = RDAmodel.Mu(i,:);
     pi = RDAmodel.Pi(i);
     h_map(:,i) = X_test*(Mu * covariance_inv)' - 0.5 * Mu * covariance_inv *Mu' + log(pi);
     
     %Classification
    [posterior, Y_predict] = max(h_map,[],2);
    
end
