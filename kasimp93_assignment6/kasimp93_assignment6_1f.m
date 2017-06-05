%Problem6_1f

clear all;
close all;
clc;

%Fixing Random Seed
s = RandStream('mt19937ar','Seed',0);

% load train and test sets
trXg = load('train.data');
trY = load('train.label');
nclasses = length(unique(trY));
teXg = load('test.data');
teY = load('test.label');

% load vocabulary and stop words
fileID = fopen('vocabulary.txt');
words = textscan(fileID,'%s','Delimiter','\n');
fileID = fopen('stoplist.txt');
stop = textscan(fileID,'%s','Delimiter','\n');
nstops =  length(stop{1,1});

% Intersection of vocabulary and stop words.
[C,ia]  = intersect(words{1,1},stop{1,1});
nwords = length(words{1,1}) - length(C)
[~,wordidcs]  = setdiff(words{1,1},stop{1,1});
wordidcs = sort(wordidcs);
wordIDs = containers.Map(wordidcs,1:length(wordidcs));
ndocs = max(trXg(:,1));

load('x_train.mat');

nTest = max(teXg(:,1));


tic;
counter = 1;

for i = 1:length(b_c)
    for j = 1:length(sigma)
        for i = 1:nclasses
            for j = i+1: nclasses
                a = find(trY == i);
                b = find(trY == j);
                X_train = x_train([a;b],:);
                Y_train = [i * ones(length(a),1) ; j * ones(length(b),1)];
                SVMmodel{counter} = svmtrain(sparse(X_train),Y_train,'kernel_function', 'rbf','autoscale', false);
                counter = counter + 1;
    end
        end
            end
end
toc

tic;
counter = 1;
for i = 1:nclasses
    for j = i+1: nclasses
        Y_predict(:,counter) = svmclassify(SVMmodel{counter},x_test);
        counter = counter + 1;
    end
end

maxVoteYPredict = mode(Y_predict,2);
CCR = sum(maxVoteYPredict == teY)/nTest
confusionmat(teY, maxVoteYPredict)
toc