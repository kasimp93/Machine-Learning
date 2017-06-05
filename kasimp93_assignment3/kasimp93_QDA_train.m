function [QDAmodel]= kasimp93_QDA_train(X_train, Y_train, numofClass)
%
% Training QDA
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
% QDAmodel : the parameters of QDA classifier which has the following fields
% QDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% QDAmodel.Sigma : D * D * numofClass array, Sigma(:,:,i) = covariance matrix of class i
% QDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i

% Write your code here:


load('data_iris.mat');
numofClass = 3;

% Dimension of Data
D = size(X_train,1);

for i = 1:numofClass
    count = 0;
for j = 1:D
   if Y_train(j) == i;
        count = count + 1;
    end
end
    % Finding mean, covariance and the Prior Probability
    Mu(i,:) = mean(X_train(Y_train==i,:));
    covariance(:,:,i) = cov(X_train(Y_train==i,:));
    pi(i) = count/D;

    %Output Model
    QDAmodel=struct('Mu',Mu,'Sigma',covariance,'pi',pi);
    save('variables');

end