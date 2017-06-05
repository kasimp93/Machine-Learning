% Waleed Tahir
% EC-503 HW05
% histogram_plots

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Histograms
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Making histograms');

% get data length
datalen  = length(Dates);

% matrix of pre-processed training
train_data = zeros(datalen,41);

% get unique districts
u_dist = char(unique(PdDistrict));

% get unique days
u_days = char('Sunday','Monday','Tuesday','Wednesday','Thursday',...
    'Friday','Saturday');

% a temp column is a sample column
hour_offset = 0;
day_offset = 24;
dist_offset = 31;

% pre processing
Dates_c = char(Dates);

temp = zeros(41,1);
tic
for i = 1:length(Category)

    % get hour of crime
    date_c = strsplit(Dates_c(i,:));
    hour_c = strsplit(char(date_c(2)),':');
    hour = str2double(char(hour_c(1)));
    
    if hour == 0
        hour = 24;
    end
    
    % update hour
    temp(hour+hour_offset) = temp(hour+hour_offset) + 1;
    
    % get day of crime
    day_num =  strmatch(char(DayOfWeek(i)),u_days);
    temp(day_num+day_offset) = temp(day_num+day_offset) + 1;
    
    % get district of crime
    dist_num = strmatch(char(PdDistrict(i)),u_dist);
    temp(dist_num+dist_offset) = temp(dist_num+dist_offset) + 1;
end
toc

hour_hist = temp(1:day_offset);
day_hist = temp(day_offset+1:dist_offset);
dist_hist = temp(dist_offset+1:end);

figure(1)
bar(hour_hist);
title('Histogram of Hours');
xlabel('hours');
ylabel('occurences');
grid on;

figure(2)
bar(day_hist);
title('Histogram of DaysOfWeek');
xlabel('days');
ylabel('occurences');
grid on;

figure(3)
bar(dist_hist);
title('Histogram of PdDIstrict');
xlabel('district');
ylabel('occurences');
grid on;
