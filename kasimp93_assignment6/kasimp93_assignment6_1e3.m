%Problem6_1e

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

% List of words without stop list.
[~,wordidcs]  = setdiff(words{1,1},stop{1,1});
wordidcs = sort(wordidcs);
wordIDs = containers.Map(wordidcs,1:length(wordidcs));
ndocs = max(train_data(:,1));

nTest = max(test_data(:,1));


tic;
counter = 1;
for i = 1:nclasses
    for j = i+1: nclasses
        a = find(label_tr == i);
        b = find(label_tr == j);
        X_train = tr_data([a;b],:);
        Y_train = [i * ones(length(a),1);  j * ones(length(b),1)];
        SVMmodel{counter} = svmtrain(sparse(X_train),Y_train,'kernel_function', 'rbf','autoscale', false);
        counter = counter + 1;
    end
end
toc

tic;
counter = 1;
for i = 1:nclasses
    for j = i+1: nclasses
        Y_predict(:,counter) = svmclassify(SVMmodel{counter},te_data);
        counter = counter + 1;
    end
end

maxVoteYPredict = mode(Y_predict,2);
CCR = sum(maxVoteYPredict == label_te)/nTest
z = confusionmat(label_te, maxVoteYPredict)

toc