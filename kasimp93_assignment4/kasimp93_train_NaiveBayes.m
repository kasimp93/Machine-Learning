function [ beta, pi ] = kasimp93_train_NaiveBayes( train_data, train_label, nclasses, nwords, alpha, wordIDs)

word_count = zeros(nclasses,1);
beta = sparse(zeros(nwords, nclasses));
doc_count = zeros(nclasses,1);
ndocs = length(train_label);

for i = 1 : size(train_data,1)
    if isKey(wordIDs,{train_data(i , 2)})
        class = train_label( train_data(i , 1) );
        id = values(wordIDs,{train_data(i , 2)});
        word_count(class) = word_count(class) + train_data(i , 3);
        beta(cell2mat(id) , class ) = beta( cell2mat(id) , class ) + train_data(i , 3);
    end
end

for k = 1: ndocs
    doc_count(train_label(k)) = doc_count(train_label(k)) + 1;
end

pi = doc_count ./ ndocs ;

for j = 1 : nclasses
    beta( : , j ) = ( beta( :, j ) + alpha(j) ) ./ ( word_count(j) + sum(alpha) );
end

end

