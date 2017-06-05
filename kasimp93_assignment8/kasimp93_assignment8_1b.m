clear all;
close all;
clc;


no_of_cluster = 3;
pts_cluster = 500 * ones(no_of_cluster,1);

[Data_circle ,label_circle] = sample_circle( no_of_cluster, pts_cluster );
[theta, rho] = cart2pol(Data_circle(:,1),Data_circle(:,2));
Dist_circle = [rho,theta];

norm_Distance_3 = (Dist_circle - kron(min(Dist_circle),ones(sum(pts_cluster),1))) ./ ...
    ( kron(max(Dist_circle),ones(sum(pts_cluster),1)) - kron(min(Dist_circle),ones(sum(pts_cluster),1)) );

figure; 
colors = {'r.','b.','g.','k.'};
centers = {'rx','bx','gx','kx'};

for j = 2:4
    rng(2);
    [idx3,Center,s] = kmeans(norm_Distance_3,j,'Distance','cityblock','Replicates',20);
    
    for i = 1: j
        h1 = subplot(1,3,j-1);
        plot(norm_Distance_3(idx3==i,1),norm_Distance_3(idx3==i,2),colors{1,i}); 
        hold on;
        
        plot(Center(i,1),Center(i,2),centers{1,i},'MarkerSize',15,'LineWidth',3);
        set(h1);
        sumd3(i) = sum(sqrt((norm_Distance_3(idx3==i,1) - Center(i, 1)).^2 + (norm_Distance_3(idx3==i,2) - Center(i, 2)).^2));
    end
    
    str = sprintf('K=%d',j);
    title(h1,str);
    
    sumd3
    
end