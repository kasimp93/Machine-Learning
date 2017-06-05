function [RDAmodel]= kasimp93_RDA_train(X_train, Y_train,gamma, numofClass)
%
% Training RDA
%
% EC 503 Learning from Data
% Gaussian Discriminant Analysis
%
% Assuming D = dimension of data
% Inputs :
% X_train : training data matrix, each row is a training data point
% Y_train : training labels for rows of X_train
% numofClass : number of classes 
%
% Assuming that the classes are labeled  from 1 to numofClass
% Output :
% RDAmodel : the parameters of RDA classifier which has the following fields
% RDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% RDAmodel.Sigmapooled : D * D  covariance matrix
% RDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:
D = size(X_train,1);

d = size(X_train,2);
covariance = zeros(d);


for i = 1:numofClass
    
    X_ti =   X_train(Y_train == i, :);
    ni = size(X_ti,1);
    Mu = sum(X_ti)/ni;
    pi = ni/D;
    covariance = covariance + (bsxfun(@minus,X_ti,Mu))' * (bsxfun(@minus,X_ti,Mu));
    RDAmodel.Mu(i,:) = Mu;
    RDAmodel.Pi(i) = pi;
end  


Sigma_RDA = covariance./(D-numofClass);
Sigmapooled = gamma * diag( diag( Sigma_RDA ) ) + (1 - gamma) * Sigma_RDA;


RDAmodel.Sigmapooled = Sigmapooled;


end