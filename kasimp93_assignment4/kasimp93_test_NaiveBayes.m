function [Y_predict, posterior] = kasimp93_test_NaiveBayes( test_data, beta, pi, nclasses)

nwords = size(beta,1);
weights = zeros(nwords,1);
posterior = zeros(nclasses,1);
wordIDs = containers.Map(1:nwords,1:nwords);

for i = 1 : size(test_data,1)
    if isKey(wordIDs,{test_data(i , 2)})
        id = values(wordIDs,{test_data(i , 2)});
        weights( cell2mat(id) ) = weights( cell2mat(id) ) + test_data(i , 3);
    end
end

weights = sparse(weights);

for i = 1:nclasses
    posterior(i) = log( pi(i) ) + weights' * log( beta(:,i) );
end

%Classification Model
[~,Y_predict] = max(posterior);
end