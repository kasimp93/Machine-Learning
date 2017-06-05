%Problem6_1b

clear all;
close all;
clc;

%Fixing Random Seed
s = RandStream('mt19937ar','Seed',0);

load('x_train1_20');
load('x_test1_20');
load('x_label1_20');
load('test_label1_20');

K = 5;
Indices = crossvalind('Kfold',xlabel,K);

pow_b = -5:15
b_c = 2.^pow_b ;

pow_s = -13:3
sigma = 2.^pow_s ;

for i = 1:length(b_c)
    for j = 1:length(sigma)
        for k = 1:K
            test = (Indices == k);
            train = ~test;
            X_train = x_train(train,:);
            X_test = x_train(test,:);
            no_train = size(X_train,1);
            no_test = size(X_test,1);
            Y_train = xlabel(train);
            Y_test = xlabel(test);
            C = b_c(i) * ones(no_train,1)
            SVMmodel = svmtrain(X_train,Y_train, 'kernel_function','rbf','rbf_sigma', sigma(j), 'boxconstraint',C, 'autoscale','false');
            Y_predict = svmclassify(SVMmodel,X_test);
            CCR(k) = sum(Y_predict == Y_test)/no_test;
        end
            avgCCR(i,j) = mean(CCR);
    end
end

figure; 
contour( log(sigma), log(b_c), avgCCR); 
colorbar;


[M,I] = max(avgCCR(:));
[cIDX, sIDX] = ind2sub(size(avgCCR),I);
C = b_c(cIDX) * ones(size(x_train,1),1);
SVMmodel = svmtrain(sparse(x_train),xlabel,'kernel_function', 'rbf', 'rbf_sigma', sigma(sIDX) ,'boxconstraint',C,'autoscale', false);
Y_predict = svmclassify(SVMmodel,x_test);
CCR = sum(Y_predict == testlabel)/size(x_test,1);