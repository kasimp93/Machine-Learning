clear all;
close all;
clc;

rng('default');

load('linear_data.mat');

%% Part (a)
% mean_x = mean(xData);
% mean_y = mean(yData);
centered_x = xData;
centered_y = yData;
w_ols = (centered_y'*centered_x)/(centered_x'*centered_x);
b_ols = mean_y - w_ols*mean_x
b(:,1)=[b_ols;w_ols];
h_ols(:,1) = w_ols * centered_x' + b_ols;
train_mae = mean(abs(centered_y - h_ols(:,1)));
train_mse = mean((centered_y - h_ols(:,1)).^2);

%% (b)
figure; 
plot(xData, h_ols(:,1)); 
hold on; 
grid on;
meth = {'ols','cauchy', 'fair','huber', 'talwar','datapoints'};
tune = [nan,2.385, 1.400, 1.345, 2.795];

for i = 2 : length(meth)-1
    b(:,i)= robustfit(xData,yData,meth{1,i},tune(i),'on');
    h_ols(:,i) = b(2,i) * xData + b(1,i);
    train_mae(i) = mean(abs(yData - h_ols(:,i)));
    train_mse(i) = mean((yData - h_ols(:,i)).^2);
    plot(xData, h_ols(:,i)); 
end

plot(xData, yData, 'x');
legend(meth, 'interpreter','latex'); 
hold off;
title('Comparison of Regression Methods');
xlabel('x');
ylabel('y');
set(gca,'TickLabelInterpreter');


%% 
meth = {'ols','cauchy', 'fair','huber', 'talwar'};
figure; 
bar(train_mae); 
grid;
title('MAE for different Regression Methods');
xlabel('Regression Method');
ylabel('MAE');


figure; 
bar(train_mse); 
grid; 
title('MSE for different Regression Methods');
xlabel('Regression Method');
ylabel('MSE');
