% Waleed Tahir
% EC-503 HW05
% likelycrime_eachdist

% data length
datalen = length(Dates);

% get unique labels
u_category = char(unique(Category));
[m,n] = size(u_category);

% label vector for training data
train_data_labels = zeros(length(Dates),1);

% get unique districts
u_dist = char(unique(PdDistrict));

likely_crime_mat = zeros(10,m);

tic
for i = 1:datalen
    
    % get district of crime
    dist_num = strmatch(char(PdDistrict(i)),u_dist);
    
    % get label of crime
    cat_num = strmatch(char(Category(i)),u_category);
    train_data_labels(i) = cat_num;
    
    
    likely_crime_mat(dist_num,cat_num) = likely_crime_mat(dist_num,cat_num)+1;
end
toc

[~,likely_crime_each_dist] = max(likely_crime_mat,[],2)