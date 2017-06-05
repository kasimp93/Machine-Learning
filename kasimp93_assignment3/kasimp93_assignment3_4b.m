clear all;
close all;
clc;

load('data_knnSimulation');
k = 10;

x = -3.5:0.1:6;
y = -3:0.1:6;
[x,y] = meshgrid(x,y);

x_y = [ x(:) , y(:) ];

n_tr = size(Xtrain,1);
n_te = size(x_y,1);

distances = sparse(zeros( n_te, n_tr ));

xte_sparse = sparse(x_y);
xtr_sparse = sparse(Xtrain);
    
distances = bsxfun( @minus, sum( xtr_sparse.^2, 2 )', 2 * xte_sparse * xtr_sparse' );

[sorted, idcs] = sort(distances,2);
Indices = idcs(:,1:k);
Distances_sorted = full(sorted(:,1:k));

classes = zeros(size(Indices));

for i = 1: size(Indices,1)
   classes(i,:) = (ytrain(Indices(i,:)))';
end

prob_c2 = sum(classes == 2 , 2)/k;
prob_c2 = reshape(prob_c2, size(x,1), size(x,2)); 

figure; 
imagesc('XData',-3.5:0.1:6,'YData',-3:0.1:6,'CData',prob_c2); 
axis([-3.5 6 -3 6]); 
colorbar;
title('Probabilities of Class 2');
xlabel('Feature 1'); 
ylabel('Feature 2');

prob_c3 = sum(classes == 3 , 2)/k;
prob_c3 = reshape(prob_c3, size(x,1), size(x,2));

figure; 
imagesc('XData',-3.5:0.1:6,'YData',-3:0.1:6,'CData',prob_c3);
axis([-3.5 6 -3 6]); 
title('Probabilities of Class 3 | K=10 ');
colorbar;
xlabel('Feature 1'); 
ylabel('Feature 2');