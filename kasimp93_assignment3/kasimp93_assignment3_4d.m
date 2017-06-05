clear all;
close all;
clc;

load('data_mnist_train.mat');
load('data_mnist_test.mat');

k = 1;


n_tr = size(X_train,1);
n_te = size(Y_test,1);

distances = sparse(zeros( n_te, n_tr ));
batchsize = 100;
xte_sparse = sparse(X_test);
xtr_sparse = sparse(X_train);

for j=1:batchsize:n_te

te = min( j + batchsize - 1, n_te );
xte = X_test ( j : te, : );
    
distances = bsxfun( @minus, sum( xtr_sparse.^2, 2 )', 2 * xte * xtr_sparse' );

[sorted, idcs] = sort(distances,2);
Indices = idcs(:,1:k);
Distances_sorted = full(sorted(:,1:k));

classes = zeros(size(Indices));

for i = 1: size(Indices,1)
   classes(i,:) = (Y_train(Indices(i,:)))';
end
    mode_classes(j:te) = mode(classes,2);
end

confusion_mat = confusionmat( Y_test, mode_classes)
ccr = sum(diag(confusion_mat))/n_te