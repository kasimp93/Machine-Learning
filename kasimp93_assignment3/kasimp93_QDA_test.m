function [Y_predict] = kasimp93_QDA_test(X_test, QDAModel, numofClass)
%
% Testing for QDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_test : test data matrix, each row is a test data point
% numofClass : number of classes 
% QDAmodel: the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance
% matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
% 
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% Y_predict predicted labels for all the testing data points in X_test

% Write your code here:


load('data_iris.mat');

%Dimension of Data
D = size(X_test,1);
count = 0;

for i= 1:numofClass
    
     Mu = QDAModel.Mu(i,:);
     covariance = QDAModel.Sigma(:,:,i);
     pi = QDAModel.pi(i);
     h_map(:,i) = diag(0.5*((bsxfun(@minus,X_test,Mu)) * inv(covariance) *(bsxfun(@minus,X_test,Mu))') + 0.5* log(det(covariance)) - log(pi));
     
end
    %Y_predict predicted labels for all the testing data points in X_test
    [posterior, Y_predict] = min(h_map,[],2);
   
    
end