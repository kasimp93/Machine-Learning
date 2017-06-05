clear all;
close all;
clc;

rng('default');

load('prostateStnd.mat');

mean_x = mean(Xtrain); 
mean_y = mean(ytrain);
mean_x= repmat(mean_x,size(Xtrain,1),1);
centered_X = Xtrain - mean_x; 
centered_Y = ytrain - mean_y;
a = 2 * sum(centered_X.^2);

%% Part (a)
ln_lambda = -5: 10;
for  i = 1: length(ln_lambda)
    lambda = exp( ln_lambda(i) );
    b = ridge(centered_Y, centered_X,lambda,0);
    
    for t = 1: 100
        for k = 1 : size(centered_X, 2)
            c(k) = 2 * sum( centered_X(:,k) .* ( centered_Y - centered_X * b(2:end)   + b(k+1)' .* centered_X(:,k)  ) );
            b(k+1) = sign(c(k)/a(k)) * max( 0, abs(c(k)/a(k)) - lambda / a(k));
        end
    end
    b_las(:,i) = b;
    yfit(:,i) = b(1) + Xtrain * b(2:end);
    ypred(:,i) = b(1) + Xtest * b(2:end);
    train_mse(i) = mean((yfit(:,i)-ytrain).^2);
    test_mse(i) = mean((ypred(:,i)-ytest).^2);
end

figure; 
plot(ln_lambda, b_las(2,:)); 
grid on; 
hold on; 
plot(ln_lambda, b_las(3,:)); 
plot(ln_lambda, b_las(4,:)); 
plot(ln_lambda, b_las(5,:)); 
plot(ln_lambda, b_las(6,:)); 
plot(ln_lambda, b_las(7,:)); 
plot(ln_lambda, b_las(8,:)); 
plot(ln_lambda, b_las(9,:));
hold off;
title('Lasso coefficients for different \ln(\lambda)');
xlabel('\ln(\lambda)');
ylabel('Ridge Coefficicents');
legend('w_1', 'w_2', 'w_3', 'w_4', 'w_5', 'w_6', 'w_7', 'w_8');

figure; plot(ln_lambda, train_mse); 
grid on; 
hold on; 
plot(ln_lambda, test_mse);  
hold off;
title('MSE for Lasso Regression with different \ln(\lambda)');
xlabel('\ln(\lambda)');
ylabel('MSE');
legend('train', 'test');

%% Part (b)
yfit_2w = b(1) + Xtrain(:,1:2) * b_las(2:3,10);
ypred_2w = b(1) + Xtest(:,1:2)  * b_las(2:3,10);
train_mse_2w = mean((yfit_2w-ytrain).^2);
test_mse_2w = mean((ypred_2w-ytest).^2);

%% Part (c)
ln_lambda = -5:10;
for i = 1: length(ln_lambda);
  b_ridge(:,i) = ridge(centered_Y, centered_X,exp(ln_lambda(i)),0);
  yfit(:,i) = b_ridge(1,i) + Xtrain * b_ridge(2:end,i);
  ypred(:,i) = b_ridge(1,i) + Xtest * b_ridge(2:end,i);
  train_mse(i) = mean((yfit(:,i)-ytrain).^2);
  test_mse(i) = mean((ypred(:,i)-ytest).^2);
end


figure;
plot(ln_lambda, b_ridge(2,:))
grid on;
hold on;
plot(ln_lambda, b_ridge(3,:))
plot(ln_lambda, b_ridge(4,:))
plot(ln_lambda, b_ridge(5,:))
plot(ln_lambda, b_ridge(6,:))
plot(ln_lambda, b_ridge(7,:))
plot(ln_lambda, b_ridge(8,:))
hold off;
title('Ridge Coefficients against ln-lambda');

figure; 
plot(ln_lambda, train_mse); 
grid on; 
hold on;
plot(ln_lambda, test_mse);  
hold off;
title('MSE for Ridge Regression with different \ln(\lambda)');
xlabel('\ln(\lambda)');
ylabel('MSE');
legend('train', 'test');