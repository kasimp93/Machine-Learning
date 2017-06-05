function [ beta, pi ] = kasimp93_train_NaiveBayes_map( train_data, train_label, no_of_classes, no_of_words, alpha)

word_count = zeros(no_of_classes,1);
beta = zeros(no_of_words, no_of_classes);
doc_count = zeros(no_of_classes,1);
ndocs = length(train_label);
wordIDs = containers.Map(1:no_of_words,1:no_of_words);

for i = 1 : size(train_data,1)
    if isKey(wordIDs,{train_data(i , 2)})
        label = train_label( train_data(i , 1) );
        id = values(wordIDs,{train_data(i , 2)});
        word_count(label) = word_count(label) + train_data(i , 3);
        beta(cell2mat(id) , label ) = beta( cell2mat(id) , label ) + train_data(i , 3);
    end
end

for k = 1: ndocs
    doc_count(train_label(k)) = doc_count(train_label(k)) + 1;
end

for j = 1 : no_of_classes
    beta( : , j ) = ( beta( :, j ) + alpha(j) ) ./ ( word_count(j) + sum(alpha) );
end

pi = doc_count ./ ndocs ;
beta = sparse(beta);
end

