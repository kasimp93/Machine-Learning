% Waleed Tahir
% EC-503 HW05
% likelyhour_eachcrime

% data length
datalen = length(Dates);

% get unique labels
u_category = char(unique(Category));
[m,n] = size(u_category);

% label vector for training data
train_data_labels = zeros(length(Dates),1);

Dates_c = char(Dates);
likely_hr_mat = zeros(m,24);

tic
for i = 1:datalen
    % get hour of crime
    date_c = strsplit(Dates_c(i,:));
    hour_c = strsplit(char(date_c(2)),':');
    hour = str2double(char(hour_c(1)));
    
    if hour == 0
        hour = 24;
    end
    
    % get label of crime
    cat_num = strmatch(char(Category(i)),u_category);
    train_data_labels(i) = cat_num;
    
    
    likely_hr_mat(cat_num,hour) = likely_hr_mat(cat_num,hour)+1;
end
toc

[~,likely_hr_each_crime] = max(likely_hr_mat,[],2)