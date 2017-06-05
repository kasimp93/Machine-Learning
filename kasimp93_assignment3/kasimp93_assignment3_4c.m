clear all;
close all;
clc;

load('data_knnSimulation');
k = 1;

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

count_1 = sum(classes == 1 , 2);
count_2 = sum(classes == 2 , 2);
count_3 = sum(classes == 3 , 2);

counts_of_all_classes = [count_1,count_2,count_3];

[l, cl_res] = max(counts_of_all_classes,[],2);
cl_res = reshape(cl_res, size(x,1), size(x,2)); 


figure; 
imagesc('XData',-3.5:0.1:6,'YData',-3:0.1:6,'CData',cl_res); 
axis([-3.5 6 -3 6]); 
colorbar;
title('Class Labels K=5');
xlabel('Feature 1'); 
ylabel('Feature 2');