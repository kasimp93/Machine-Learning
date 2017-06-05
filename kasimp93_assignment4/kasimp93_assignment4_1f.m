%Problem 4.1f
clear;
close all;
clc;

% Load train and test set
train_data = load('train.data');
train_label = load('train.label');
test_data = load('test.data');
test_label = load('test.label');
no_of_classes = 20;

fileID = fopen('vocabulary.txt','r');
words = textscan(fileID,'%s');
no_of_words =  length(words{1,1});


[C1_b4,~,~] = unique(data_tr(:,1));
[C2_b4,~,~] = unique(test_data(:,1));


% read stop words and vocabulary
file1 = fopen('stoplist.txt');
file2 = fopen('vocabulary.txt');
stoplist = textscan(file1,'%s');
stoplist = stoplist{1,1};
vocabulary = textscan(file2,'%s');
vocabulary = vocabulary{1,1};
clear file1 file2 

% get inidices of stop words in vocab, data_tr, data_te
stop_indices = find(ismember(vocabulary,stoplist)==1);
tr_stop_indices = ismember(train_data(:,2), stop_indices);
te_stop_indices = ismember(test_data(:,2), stop_indices);

% remove stop words from vocab, data_tr, data_te
vocabulary(stop_indices) = [];
train_datat(tr_stop_indices,:) = [];
test_data(te_stop_indices,:) = [];

% compute average word count
word_count_trdoc = zeros(1,num_tr_docs);
for i = 1 : length(train_data(:,1))
    word_count_trdoc(train_data(i,1)) = word_count_trdoc(train_data(i,1))+train_data(i,3);
end
avg_word_count_trdoc = mean(word_count_trdoc);

word_count_tedoc = zeros(1,num_te_docs);
for i = 1 : length(test_data(:,1))
    word_count_tedoc(test_data(i,1)) = word_count_tedoc(test_data(i,1))+test_data(i,3);
end
avg_word_count_tedoc = mean(word_count_tedoc);

% updated doc counts
num_te_docs = length(unique(test_data(:,1)));
num_tr_docs = length(unique(train_data(:,1)));

[C1_after,~,~] = unique(train_data(:,1));
[C2_after,~,~] = unique(test_data(:,1));

% remove labels of no longer relevant docs
rm_doc_index_tr = find (ismember(C1_b4,C1_after) == 0);
rm_doc_index_te = find (ismember(C2_b4,C2_after) == 0);
label_tr(rm_doc_index_tr) = [];
label_te(rm_doc_index_te) = [];


% train Naive Bayes
alpha = 1/no_of_words * ones(no_of_classes, 1);
[ beta, pi ] = kasimp93_train_NaiveBayes_map( train_data, train_label, no_of_classes, no_of_words,alpha);

% Among the W × 20 estimated parameters (beta_{w,c}'s), how many of them are zero?
nzeros_beta = sum(beta(:)==0)

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
testCCR = testCCR*100
