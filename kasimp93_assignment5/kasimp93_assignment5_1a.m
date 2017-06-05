clear all;
close all;
clc;

%loading the data
load('data_SFcrime_train.mat');
load('data_SFcrime_test.mat');
no_tr = size(DayOfWeek,1);
no_te = size(DayOfWeek_test,1);

%Initializing the values of variables
day_train = zeros(size(DayOfWeek,1),7);
day_test = zeros(size(DayOfWeek_test,1),7);
pd_train = zeros(size(PdDistrict,1),10);
pd_test = zeros(size(PdDistrict_test,1),10);
hour_train = zeros(size(hour(Dates),1),24);
hour_test = zeros(size(hour(Dates_test),1),24);
hour_q = zeros(no_tr,1);
day_q = zeros(no_tr,1);
pd_q = zeros(no_tr,1);

%Assigning keys for containers
key_day = {'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'};
key_pd = {'BAYVIEW', 'CENTRAL','INGLESIDE','MISSION','NORTHERN','PARK','RICHMOND','SOUTHERN','TARAVAL','TENDERLOIN'};
key_hour = {'0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23'};

%Assigning keys for containers
values_day = 1:7;
values_pd = 1:10;
values_hour = 1:24;

%Mapping values and a unique key for each value.
day = containers.Map(key_day,values_day);
pd = containers.Map(key_pd,values_pd);
hour_map = containers.Map(key_hour,values_hour);

categories = unique(Category);

for n = 1:size(categories,1)
KeyCat{n,1} = categories{n,1};
valueCat(n) = n;
end
Cat = containers.Map(KeyCat,valueCat);

hour_cat = zeros(24,n);
pd_cat = zeros(10,n);
day_cat = zeros(7,n);


for i= 1:no_tr
    
hour_q(i) = hour(Dates{i,1});
hour_train(i,hour_q(i)+1) = 1;
crime = cell2mat(values(Cat,Category(i,1)));
hour_cat(hour_q(i)+1,crime) = hour_cat(hour_q(i)+1,crime)+1;

day_q(i) = cell2mat(values(day,DayOfWeek(i,1)));
day_train(i,day_q(i))=1;


pd_q(i) = cell2mat(values(pd,PdDistrict(i,1)));
pd_train(i,pd_q(i))=1 ;
pd_cat(pd_q(i) , crime) = pd_cat(pd_q(i) , crime) + 1;


end

train_data = [hour_train, day_train, pd_train];
save('train_data.mat','train_data');

[~,lik_crime_hr] = max(hour_cat);
lik_crime_hr = lik_crime_hr - 1;

[~,lik_crime_dist] = max(pd_cat,[],2);

figure; 
histogram(hour_q); 
title('Hours of Day'); 
axis([0 23 -inf +inf]);

figure; 
histogram(day_q); 
title('Days of Week');
axis([1 7 -inf +inf]);

figure; 
histogram(pd_q); 
title('Police Department Districts'); 
axis([1 10 -inf +inf]);

for i= 1:no_te
    
hour_r = hour(Dates_test{i,1});
hour_test(i,hour_r+1) = 1;
crime = cell2mat(values(Cat,Category(i,1)));
hour_cat(hour_r+1,crime) = hour_cat(hour_r+1,crime)+1;

day_r = cell2mat(values(day,DayOfWeek(i,1)));
day_test(i,day_r)=1;
 
pd_r = cell2mat(values(pd,PdDistrict(i,1)));
pd_test(i,pd_r)=1 ;
pd_cat(pd_r , crime) = pd_cat(pd_r , crime) + 1;


end

test_data = [hour_test, day_test, pd_test];
save('test_data.mat','test_data');