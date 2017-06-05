clear all;
close all;
clc;

load('data_knnSimulation');

figure;
gscatter(Xtrain(:,1),Xtrain(:,2),ytrain,'rgb')
grid on;
title ('Scatter Plot of All the Training Data');
xlabel('Feature 1');
ylabel('Feature 2');