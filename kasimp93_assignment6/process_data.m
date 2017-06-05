% Waleed Tahir
% EC-503 HW06
% Data PreProcessing
clear;
clc;

% read data from files
data_tr = load('data\\train.data');
data_te = load('data\\test.data');
label_tr = load('data\\train.label');
label_te = load('data\\test.label');
num_te_docs = length(unique(data_te(:,1)));
num_tr_docs = length(unique(data_tr(:,1)));

wrdIDs_total = unique([unique(data_te(:,2));unique(data_tr(:,2))]);
unique_wrdIDs_total = length(wrdIDs_total);

[C1_b4,~,~] = unique(data_tr(:,1));
[C2_b4,~,~] = unique(data_te(:,1));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove stop words here from training and testing sets

% read stop words and vocabulary
file1 = fopen('data\\stoplist.txt');
file2 = fopen('data\\vocabulary.txt');
stoplist = textscan(file1,'%s');
stoplist = stoplist{1,1};
vocabulary = textscan(file2,'%s');
vocabulary = vocabulary{1,1};
clear file1 file2 

% get inidices of stop words in vocab, data_tr, data_te
stop_indices = find(ismember(vocabulary,stoplist)==1);
tr_stop_indices = ismember(data_tr(:,2), stop_indices);
te_stop_indices = ismember(data_te(:,2), stop_indices);
% remove stop words from vocab, data_tr, data_te
vocabulary(stop_indices) = [];
data_tr(tr_stop_indices,:) = [];
data_te(te_stop_indices,:) = [];

% unique word ids after stop words removed
wrdIDs_after = unique([unique(data_te(:,2));unique(data_tr(:,2))]);
unique_wrdIDs_after = length(wrdIDs_after);

% updated doc counts
num_te_docs = length(unique(data_te(:,1)));
num_tr_docs = length(unique(data_tr(:,1)));

[C1_after,~,~] = unique(data_tr(:,1));
[C2_after,~,~] = unique(data_te(:,1));

% remove labels of no longer relevant docs
rm_doc_index_tr = find (ismember(C1_b4,C1_after) == 0);
rm_doc_index_te = find (ismember(C2_b4,C2_after) == 0);
label_tr(rm_doc_index_tr) = [];
label_te(rm_doc_index_te) = [];

% fix docIDs after removal of 1 doc that vanished with stoplist words
data_tr(data_tr(:,1)>rm_doc_index_tr,1) = data_tr(data_tr(:,1)>rm_doc_index_tr,1) - 1;
data_te(data_te(:,1)>rm_doc_index_te,1) = data_te(data_te(:,1)>rm_doc_index_te,1) - 1;

%%% Mapping
keySet =   sort(wrdIDs_after)';
valueSet = 1:unique_wrdIDs_after;
mapObj = containers.Map(keySet,valueSet);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate training set for processing
tr_data = zeros(num_tr_docs,unique_wrdIDs_after);
te_data = zeros(num_te_docs,unique_wrdIDs_after);

tic
for i = 1:num_tr_docs
    one_doc = data_tr(data_tr(:,1)==i,:); 
    tr_data(i,cell2mat(values(mapObj,num2cell((one_doc(:,2)))))) = one_doc(:,3);
end
toc
tic
for i = 1:num_te_docs
    one_doc = data_te(data_te(:,1)==i,:); 
    te_data(i,cell2mat(values(mapObj,num2cell((one_doc(:,2)))))) = one_doc(:,3);
end
toc

% normalize data
num_trwords = repmat(sum(tr_data,2),1,size(tr_data,2));
tr_data = tr_data./num_trwords;
num_tewords = repmat(sum(te_data,2),1,size(te_data,2));
te_data = te_data./num_tewords;

clearvars -except tr_data te_data label_tr label_te
save('processed_data_allclasses','-v7.3');

% 2 class training set
idx_tr_1 = find((label_tr == 1));
idx_tr_20 = find((label_tr == 20));

tr_data_2class = [tr_data(idx_tr_1,:);tr_data(idx_tr_20,:)];
label_tr_2class = [label_tr(label_tr == 1);label_tr(label_tr == 20)];

% 2 class test set
idx_te_1 = find((label_te == 1));
idx_te_20 = find((label_te == 20));

te_data_2class = [te_data(idx_te_1,:);te_data(idx_te_20,:)];
label_te_2class = [label_te(label_te == 1);label_te(label_te == 20)];

clearvars -except tr_data_2class te_data_2class label_tr_2class label_te_2class
save('processed_data_2class','-v7.3');



