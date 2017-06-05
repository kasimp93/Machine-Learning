function [LDAmodel] = LDA_train(X_train, Y_train, numofClass)
%
% Training LDA
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
% LDAmodel : the parameters of LDA classifier which has the following fields
% LDAmodel.Mu : numofClass * D matrix, i-th row = mean vector of class i
% LDAmodel.Sigmapooled : D * D  covariance matrix
% LDAmodel.Pi : numofClass * 1 vector, Pi(i) = prior probability of class i
%

% Write your codes here:


load('data_iris.mat');

D = size(X_train,1);
count = 0;
d = size(X_train,2);
covariance = zeros(d);


for i = 1:numofClass
    count = 0;
for j = 1:D
   if Y_train(j) == i;
        count = count + 1;
    end
end
    %Finding Mean,Covairance and Prior Probability
    Mu(i,:) = mean(X_train(Y_train==i,:));
    pi(i) = count/D;
    diff = X_train(Y_train==i,:) - repmat(Mu(i,:),count,1);
    covariance = covariance + diff'*diff;
  
end

%Finding Sigmapooles
Sigmapooled = covariance/(D-numofClass);

%Saving Model
LDAmodel.Mu = Mu;
LDAmodel.Pi = pi;
LDAmodel.Sigmapooled = Sigmapooled;


Mdl = fitcdiscr(X_train,Y_train);
LDAModel=struct('Mu',Mu,'Sigmapooled',covariance,'pi',pi);
save('variables');
end