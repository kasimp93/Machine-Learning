clear all;
close all;
clc;

%Fixing Random Seed
s = RandStream('mt19937ar','Seed',0);

%Loading Data
load('train_data');
load('data_SFcrime_train');
Categories = unique(Category);
%Mapping Data
for m = 1:size(Categories,1)
KeyCat{m,1} = Categories{m,1};
valueCat(m) = m;
end
Ca_t = containers.Map(KeyCat,valueCat);

%Partioning Data into Test and Train Set
n= Category;
c = cvpartition(n,'HoldOut',0.4);
train = training(c,1);
test = test(c,1);
X_train = train_data(train,:);
X_test = train_data(test,:);
no_tr = size(X_train,1);
no_te = size(X_test,1);
Y_train = Category(train);
Y_test = Category(test);

%Initialization for logistic Regression
lamda = 1000;
eta = 1e-5;
tmax = 1000;

%Initialization for logistic Regression
w_k = zeros(m,41);
del_F = zeros(m,41);
f = zeros(tmax,1);
loss = zeros(no_te,1);
log_loss = zeros(tmax,1);
CCR = zeros(tmax,1);
ones_y = zeros(no_tr, m);

for class = 1:m
  idcs = find( strcmp(Y_train,Categories{class,1}) );
  ones_y(idcs,class) = 1;
end

%Gradient Descent
 for t = 1:tmax
     f2 = 0;
     w_l2 = 0;
   dotprod = X_train * w_k';
   den = sum(exp(dotprod),2);
    for k=1:m
        num = exp(X_train * w_k(k,:)' );
        delF(k,:) = (num ./ den -  ones_y(:,k))' * X_train;
        f2 = f2 +  (ones_y(:,class)' * X_train ) * w_k(k,:)';
        w_l2 = w_l2 + norm(w_k(k,:),2)^2;
    end
    
    f(t) = sum( log(den) ) - f2 + lamda/2 * w_l2;
    delF = delF + lamda * w_k;
    w_k = w_k - eta * delF;
    
 
 
     % log-loss function
    loss_num = exp(X_test * w_k');
    test_labels = cell2mat( values( Ca_t, Y_test ) );
    loss_den = sum( loss_num, 2 );
    for j = 1:no_te
        loss(j) = loss_num( j, test_labels(j)) ./loss_den(j);
        if loss(j) < 1e-10
            loss(j) = 1e-10;
        end
    end
    log_loss(t) = -1/no_te * sum( log( loss ) );
    [~,Y_predict] = max(loss_num,[],2);
    CCR(t) = sum( Y_predict == test_labels) / no_te ;
 end

figure; 
plot(f); 
title('Logisitic Regression'); 
grid;
xlabel('Iterations');
ylabel('Objective function f(\theta_t)');

figure; 
plot(log_loss); 
title('test log-loss'); 
grid;
xlabel('Iterations');
ylabel('log-loss');

figure; 
plot(CCR); 
title('Correct Classification Rate (CCR)'); 
grid;
xlabel('Iterations');
ylabel('CCR');