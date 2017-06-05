clear;
close all;
clc;

% Load train and test set
train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');
no_of_classes = 20;

no_of_words =  length(unique(train_data(:,2)));

% train Naive Bayes
[ beta, pi ] = kasimp93_train_NaiveBayes_mle( train_data, train_label, no_of_classes, no_of_words);

% Among the W × 20 estimated parameters (beta_{w,c}'s), how many of them are zero?
nzeros_beta = sum(beta(:)~=0)

% Split test data for each document
nt_docs = test_data(end,1);
counter = 0;
Y_predict = zeros(nt_docs,1);
parfor i = 1: nt_docs
    testX = [];
    testX = test_data(test_data(:,1) == i,:);
    % test Naive Bayes
    [Y_predict(i), posterior] = kasimp93_test_NaiveBayes( testX, beta, pi, no_of_classes );

    if sum(isinf(posterior)) == no_of_classes || sum(isnan(posterior)) == no_of_classes
        counter = counter + 1;
    end
end

% For some test documents, P(Y = c|x) = 0 for all c = 1, . . . , 20. What is the total number of such
% test documents?
Test_Documents_zero = counter ;

% The test CCR
testCCR = sum(Y_predict == test_label) / nt_docs