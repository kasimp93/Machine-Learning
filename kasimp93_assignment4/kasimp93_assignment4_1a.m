clear all;
close all;
clc;

%loading Data
train_data = load('train.data');
test_data = load('test.data');
train_label = load('train.label');
test_label = load('test.label');

%Unique words that appear in test, training and the entire data set
train_words = unique(train_data(:,2));
test_words = unique(test_data(:,2));
total_words = unique([train_words;test_words]);

train_unique = length(train_words);
test_unique = length(test_words);
total_words_unique = length(total_words);

train_ndocs = max(train_data(:,1));
test_ndocs = max(test_data(:,1));

train_length = zeros(train_ndocs,1);
test_length = zeros(test_ndocs,1);

for i = 1 : size(test_data,1)
    test_length(test_data(i,1),1) = test_length(test_data(i,1),1) + test_data(i , 3);
end

for i = 1 : size(train_data,1)
    train_length(train_data(i,1),1) = train_length(train_data(i,1),1) + train_data(i , 3);
end

train_average = mean(train_length);
test_average = mean(test_length);

% Total number of unique words appearing in the test set, but not in the training set.
X = setdiff(test_words,train_words);
Unique = length(X);