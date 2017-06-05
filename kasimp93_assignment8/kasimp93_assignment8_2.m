clear all;
close all;
clc;

% rng(2);
load('BostonListing');
Data1 = [longitude,latitude];
classes = unique(neighbourhood);
gtruth = zeros(size(neighbourhood));
for i = 1 : length(classes)
    idcs = find(ismember(neighbourhood, classes{i,1}));
    gtruth(idcs) = i;
end
sig = 0.01;
S1 = exp( -pdist2(Data1,Data1,'squaredeuclidean')./(2*sig^2) );
W1 = S1;
D1 = diag(sum(W1,2));

%% Spectral Clustering
L1 = D1 - W1;
L_sym1 = D1^(-1/2)*L1*D1^(-1/2);
[vLsym1,eLsym1] = eig(L_sym1);
[seLsym1, idcsLsym1] = sort(diag(eLsym1));
svLsym1 = vLsym1(:,idcsLsym1);

purity = zeros(25,1);
figure;
for k = 1:25
    %SC-3
    Vsym1 = normr(svLsym1(:,1:k));
    rng(2);
    [idxLsym1] = kmeans(Vsym1,k);
    for i = 1: k
        gt_k = gtruth(idxLsym1==i);
        [M,F] = mode(gt_k);
        purity(k) = purity(k) + F;
        if k == 5
            plot(longitude(idxLsym1==i),latitude(idxLsym1==i),'.','MarkerSize',5); hold on;
        end
    end
    purity(k) = purity(k)/length(gtruth);
end

plot_google_map;
figure; 
plot(purity,'x--'); 
grid on;
title('Clusters Purity');
xlabel('k');
ylabel('Purity');
