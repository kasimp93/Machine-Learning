%Problem6_1a

clear all;
close all;
clc;

%Fixing Random Seed
s = RandStream('mt19937ar','Seed',0);


train_data = load('train.data');
test_data = load('test.data');
train_label = load('train.label');
test_label = load('test.label');
classes = unique(train_data);
no_classes = length(classes);

% load vocabulary and stop words
fileID = fopen('vocabulary.txt','r');
words = textscan(fileID,'%s','Delimiter','\n');
fileID = fopen('stoplist.txt','r');
stop = textscan(fileID,'%s');
nstops = length(stop{1,1});

%Removing Stop Words
[C]  = intersect(words{1,1},stop{1,1});
[w,ia]  = setdiff(words{1,1},stop{1,1},'stable');
nwords = length(ia);
wordIDs = containers.Map(ia,1:nwords);

a = find(train_label == 1);
b = find(train_label == 20);
indices_train = [a;b];
ndocs = length(indices_train);
docIDs = containers.Map(indices_train,1:ndocs);
nj = zeros(ndocs,1);
x_train = zeros(ndocs, nwords);

for i = 1 : size(train_data,1)
    if isKey(wordIDs,{train_data(i , 2)}) && isKey(docIDs,{train_data(i , 1)})
        id = values(wordIDs,{train_data(i , 2)});
        doc = values(docIDs,{train_data(i , 1)});
        x_train(cell2mat(doc), cell2mat(id)) = train_data(i , 3);
        nj(cell2mat(doc)) = nj(cell2mat(doc)) + train_data(i , 3);
    end
end

for j = 1:ndocs
    x_train(j,:) = x_train(j,:) ./ nj(j);
    if j <= length(a)
        xlabel(j,1) = 1;
    else
        xlabel(j,1) = 20;
    end
end

K = 5;
indices = crossvalind('Kfold',xlabel,K);

power = -5:15;
c = 2.^power;

for i = 1:length(c)
    for k = 1 : K
        test = (indices == k);
        train = ~test;
        X_train = x_train(train,:);
        X_test = x_train(test,:);
        ntrain = size(X_train,1);
        ntest = size(X_test,1);
        Y_train = xlabel(train);
        Y_test = xlabel(test);
        C = c(i) * ones(ntrain,1);
        SVMmodel = svmtrain(sparse(X_train),Y_train,'boxconstraint',C,'autoscale', false);
        Y_predict = svmclassify(SVMmodel,X_test);
        CCR(k) = sum(Y_predict == Y_test)/ntest;
    end
    avgCCR(i) = mean(CCR);
end

figure; plot(log(c), avgCCR);

a = find(test_label == 1);
b = find(test_label == 20);
indices_test = [a;b];
tndocs = length(indices_test);
tdocIDs = containers.Map(indices_test,1:tndocs);
x_test = zeros(nTest, nwords);

for i = 1 : size(test_data,1)
    if isKey(wordIDs,{test_data(i , 2)}) && isKey(tdocIDs,{test_data(i , 1)})
        id = values(wordIDs,{test_data(i , 2)});
        doc = values(tdocIDs,{test_data(i , 1)});
        x_test(cell2mat(doc), cell2mat(id)) = test_data(i , 3);
        nj(cell2mat(doc)) = nj(cell2mat(doc)) + test_data(i , 3);
    end
end

for j = 1:tndocs
    x_test(j,:) = x_test(j,:) ./ nj(j);
    if j <= length(a)
        testlabel(j,1) = 1;
    else
        testlabel(j,1) = 20;
    end
end
        [ccrPpt, cOptIDX] = max(avgCCR);
        C = c(cOptIDX) * ones(ndocs,1);
        SVMmodel = svmtrain(sparse(x_train),xlabel,'boxconstraint',C,'autoscale', false);
        Y_predict = svmclassify(SVMmodel,x_test);
        CCR = sum(Y_predict == testlabel)/tndocs;
        