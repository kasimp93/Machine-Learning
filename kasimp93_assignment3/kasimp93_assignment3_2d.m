clear all;
close all;
clc;

% s = RandStream('mt19937ar','Seed',0)


stream = RandStream.create('mrg32k3a','NumStreams',1);
RandStream.setGlobalStream(stream);
load('data_cancer.mat');

gamma = [0.1:0.05:1];
L = length(gamma);
CCR_LDA_train=zeros(L,1);
CCR_LDA_test=zeros(L,1);
 
c = cvpartition(Y,'HoldOut',0.306);
 train = training(c,1);
 test = ~train;
 X_train = X(train,:);
 X_test = X(test,:);
 Y = Y + 1;
 Y_train = Y(train);
 Y_test = Y(test);
 numofClass = length(unique(Y));
 %number of training samples
 D_train = length(Y_train);
 D_test = length(Y_test);
 
 for i= 1:L
 [RDAmodel]= kasimp93_RDA_train(X_train, Y_train,gamma(i), numofClass);
 %[RDAmodel , Rec_Cond(i)] = kasimp93_RDA_train( X_train, Y_train , gamma(i) , numofClass );
 Y_predict_train = kasimp93_RDA_test(X_train, RDAmodel, numofClass);
 
 CCR_LDA_train(i) = sum(Y_train == Y_predict_train)/D_train; 

 Y_predict_test = kasimp93_RDA_test(X_test, RDAmodel, numofClass);
 CCR_LDA_test(i) = sum(Y_test == Y_predict_test)/D_test; 
 
 end
 
 % Showing Accuracy
figure; 
plot( gamma , CCR_LDA_train, gamma, CCR_LDA_test); 
legend('Training', 'Testing'); 
title('Regualized-LDA Performance');
xlabel('Gamma');
ylabel('Accuracy');
grid;
 