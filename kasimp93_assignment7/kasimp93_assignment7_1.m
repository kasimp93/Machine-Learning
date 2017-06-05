clear all;
close all;
clc;

%% Part a
load('housing_data.mat');
tree = fitrtree(Xtrain,ytrain,'PredictorNames',feature_names, 'ResponseName',cell2mat(output_name),'MinLeafSize',20);
view(tree,'Mode','graph');

%% Part b
test = [5, 18, 2.31, 1, 0.5440, 2, 64, 3.7, 1, 300, 15, 390, 10];
predicted = predict(tree,test);

%% Part c
for i = 1 : 25
  tree = fitrtree(Xtrain,ytrain,'PredictorNames',feature_names, 'ResponseName',cell2mat(output_name),'MinLeafSize',i ); 
  y_fit = predict(tree,Xtrain);
  train_mae(i) = mean(abs(y_fit-ytrain));
  
  y_predict = predict(tree,Xtest);
  test_mae(i) = mean(abs(y_predict-ytest));
end

figure;
plot(1:25, train_mae, 1:25, test_mae);
grid;
legend('Training MAE', 'Test MAE');
title('MAE of Test and Training Data');
xlabel('Minimum Observations per leaf');
ylabel('MAE');