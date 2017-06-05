%Problem6_1c and d

clear all;
close all;
clc;

%Fixing Random Seed
s = RandStream('mt19937ar','Seed',0);

% load train and test sets
train_data = load('train.data');
train_label = load('train.label');
nclasses = length(unique(train_label));
test_data = load('test.data');
test_label = load('test.label');

% load vocabulary and stop words
fileID = fopen('vocabulary.txt');
words = textscan(fileID,'%s','Delimiter','\n');
fileID = fopen('stoplist.txt');
stop = textscan(fileID,'%s','Delimiter','\n');
nstops =  length(stop{1,1});


[C,ia]  = intersect(words{1,1},stop{1,1});
nwords = length(words{1,1}) - length(C)
[~,wordidcs]  = setdiff(words{1,1},stop{1,1});
wordidcs = sort(wordidcs);
wordIDs = containers.Map(wordidcs,1:length(wordidcs));
ndocs = max(train_data(:,1));


for j = 1:ndocs
    if train_label(j) == 17
        x_label(j,1) = 1;
    else
        x_label(j,1) = -1;
    end
end

load('x_train.mat');

% 5-folds division
K = 5;
indices = crossvalind('Kfold',x_label,K);

pow = -5:15;
c = 2.^pow;

for i = 1:length(c)
    for k = 1 : K
        test = (indices == k);
        train = ~test;
        X_train = x_train(train,:);
        X_test = x_train(test,:);
        ntrain = size(X_train,1);
        ntest = size(X_test,1);
        Y_train = x_label(train);
        Y_test = x_label(test);
        SVMmodel = svmtrain(X_train,Y_train,'boxconstraint',c(i),'autoscale', false,'kernelcachelimit', ntrain);
        Y_predict = svmclassify(SVMmodel,X_test);
        CCR(k) = sum(Y_predict == Y_test)/ntest;
        conf = confusionmat(Y_test, Y_predict);
        precision(k) = conf(1,1)/(conf(1,1) + conf(2,1) );
        recall(k) = conf(1,1)/(conf(1,1) + conf(1,2) );
        fscore(k) = 2 * precision(k) * recall(k)/( precision(k) + recall(k) );
    end
    avgCCR(i) = mean(CCR);
    avgPrecision(i) = mean(precision);
    avgRecall(i) = mean(recall);
    avgFScore(i) = mean(fscore);
end

figure; 
plot(log(c),avgCCR); 
grid;
title('Average CCR for Linear SVM (Class 17 vs. All Others)');
xlabel('\log(c)');
ylabel('CCR');

figure; 
plot(log(c), avgPrecision, log(c), avgRecall, log(c), avgFScore); 
grid;
legend('Average Precision','Average Recall','Average F-Score');
title('Linear SVM Performance (Class 17 vs. All Others)');
xlabel('\log(c)');
ylabel('Performance Metric');

% preparing test data feature set
nTest = length(test_label);

for j = 1:nTest

    if test_label(j) == 17
        testlabel(j,1) = 1;
    else
        testlabel(j,1) =-1;
    end
end
save('x_test.mat','x_test');
load('x_test.mat');

% training with C*, found using CCR
[ccrPpt, cOptIDX] = max(avgCCR);
c_ccr = c(cOptIDX)
SVMmodel = svmtrain(x_train,x_label,'boxconstraint',c_ccr,'autoscale', false,'kernelcachelimit', ndocs);

% classifying test data
Y_predict = svmclassify(SVMmodel,x_test);
CCR_CCR = sum(Y_predict == testlabel)/nTest
ConfusionMat_CCR = confusionmat(testlabel, Y_predict)

% training with C*, found using Recall
[ccrPpt, cOptIDX] = max(avgRecall);
c_recall = c(cOptIDX)
SVMmodel = svmtrain(x_train,x_label,'boxconstraint',c_recall,'autoscale', false,'kernelcachelimit', ndocs);

% classifying test data
Y_predict = svmclassify(SVMmodel,x_test);
CCR_Recall = sum(Y_predict == testlabel)/nTest
ConfusionMat_Recall = confusionmat(testlabel, Y_predict)

% training with C*, found using F-Score
[ccrPpt, cOptIDX] = max(avgFScore);
c_fscore = c(cOptIDX)
SVMmodel = svmtrain(x_train,x_label,'boxconstraint',c_fscore,'autoscale', false,'kernelcachelimit', ndocs);

% classifying test data
Y_predict = svmclassify(SVMmodel,x_test);
CCR_FScore = sum(Y_predict == testlabel)/nTest
ConfusionMat_FScore = confusionmat(testlabel, Y_predict)