clear all;
close all;
clc;

no_cluster = 3;
pts_cluster = 500 * ones(no_cluster,1);

[Data_circle ,label_circle] = sample_circle( no_cluster, pts_cluster );
[Data_spiral ,label_spiral] = sample_spiral( no_cluster, pts_cluster );

figure; 
colors = {'r.','b.','g.','k.'};
centers = {'rx','bx','gx','kx'};

for i = 2:4
   
    rng(2);
    [index_1,Center_1,sum_circle] = kmeans(Data_circle,i,'Distance','sqeuclidean','Replicates',20);
    rng(2);
    [index_2,Center_2,sum_spiral] = kmeans(Data_spiral,i,'Distance','sqeuclidean','Replicates',20);
   
    sum_circle
    
    sum_spiral
    
    for j = 1: i
        
        h1 = subplot(3,2,2*(i-1)-1);
        plot(Data_circle(index_1==j,1),Data_circle(index_1==j,2),colors{1,j}); 
        hold on;
        legend('Cluster','Centroid');
        
        plot(Center_1(j,1),Center_1(j,2),centers{1,j},'MarkerSize',15,'LineWidth',3);
        h2 = subplot(3,2,2*(i-1));
        set(h1);
        legend('Cluster','Centroid');
        
        plot(Data_spiral(index_2==j,1),Data_spiral(index_2==j,2),colors{1,j}); 
        hold on;
        legend('Cluster','Centroid');
        
        plot(Center_2(j,1),Center_2(j,2),centers{1,j},'MarkerSize',15,'LineWidth',3);
        set(h2);
        legend('Cluster','Centroid');
    end
    
    str = sprintf('K=%d',i);
    title(h1,str);
    title(h2,str);    
end
