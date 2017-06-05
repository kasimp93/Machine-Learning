clear all;
close all;
clc;

rng('default');

load('quad_data.mat');

%% (a)
r_x = [];
r_x_test = [];
figure;
plot(xtrain, ytrain, 'o'); 
grid on; 
hold on;
for d = 1:14
    r_x = [r_x, xtrain.^d];
    b = ridge(ytrain,r_x,0,0);
    yfit(:,d) = b(1) + r_x * b(2:end);
    if d == 2 || d == 6 || d == 10 || d == 14
        plot(xtrain, yfit(:,d));
    end
    r_x_test = [r_x_test, xtest.^d];
    ypredict(:,d) = b(1) + r_x_test * b(2:end);
    MSE_train(d) = mean((yfit(:,d) - ytrain).^2); 
    MSE_test(d) = mean((ypredict(:,d) - ytest ).^2); 
end
hold off;
legend('datapoints','degree=2', 'degree=6', 'degree=10', 'degree=14'); 
hold off;
title('Regression with different degree polynomials');
xlabel('x');
ylabel('y');

figure; 
plot(MSE_train); 
grid on; 
hold on; 
plot(MSE_test);  
hold off;
title('MSE for Regression with different degrees of polynomials');
xlabel('d');
ylabel('MSE');
legend('Train', 'Test');

%% Part (b)
ln_lambda = -25:5;
for i = 1:length(ln_lambda)
    b_ridge = ridge(ytrain,r_x(:,1:10),exp(ln_lambda(i)),0);
    yfit_ridge(:,i) = b_ridge(1) + r_x(:,1:10) * b_ridge(2:end);
    ypredict_ridge(:,i) = b_ridge(1) + r_x_test(:,1:10) * b_ridge(2:end);
    MSE_train_ridge(i) = mean((yfit_ridge(:,i) - ytrain).^2); 
    MSE_test_ridge(i) = mean((ypredict_ridge(:,i) - ytest ).^2); 
end
figure; 
plot(ln_lambda, MSE_train_ridge); 
grid on; 
hold on; 
plot(ln_lambda, MSE_test_ridge);  
hold off;
title('MSE for Ridge Regression with different 1\ln 1\lambda)');
xlabel('1/ln 1/lambda)');
ylabel('MSE');
legend('train', 'test');

[m, idx] = min(MSE_test_ridge)
figure; 
plot(xtest, ytest, 'x'); 
hold on; 
grid on;
plot(xtest, ypredict(:,10));
plot(xtest, ypredict_ridge(:,idx));
legend('datapoints', 'ols', 'ridge');
title('Prediction with OLS and Ridge methods');
xlabel('x');
ylabel('y');

%% (c)
ln_lambda = -25:5;
clear b_ridge;
for i = 1: length(ln_lambda);
  b_ridge(:,i) = ridge(ytrain, r_x(:,1:4),exp(ln_lambda(i)),0);
end

figure; plot(ln_lambda, b_ridge(1,:)); 
grid on; 
hold on;
plot(ln_lambda, b_ridge(2,:)); 
plot(ln_lambda, b_ridge(3,:)); 
plot(ln_lambda, b_ridge(4,:)); 
plot(ln_lambda, b_ridge(5,:)); 
hold off;
title('Ridge coefficients for different \ln(\lambda)');
xlabel('\ln(\lambda)');
ylabel('Ridge Coefficicents');
legend('w_0', 'w_1', 'w_2', 'w_3', 'w_4');
