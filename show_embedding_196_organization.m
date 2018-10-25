
close all
clc
clear all
%% load embedding

rng('default');

datatable = readtable('list_image_single.csv');

%% load validation image filenames

fs = textread('list_train_196_organization.txt', '%s');

ID = [];
for j=1:length(fs)
    id = []; id = strfind(fs{j}, '/');
    id_image = []; id_image = str2num(fs{j}(id(end)+1:end-4));
    
    ID = [ID; find(datatable.ImageID == id_image)];
end

% fs = textread('list_train_196.txt', '%s');

N = length(fs);


%%
% % parameter for SSC
% lambda = 0.01; 
% r=0; s=1; alpha = 100;
% affine = true; outlier = false;
% rho = 0.7;
% 
% K = 2;
% % SSC running without position info
% 
% [grps,CMat, CKSym] = SSC_new(data.x_train_encoded',r,affine,alpha,outlier,rho,K);
% 
% imagesc(CKSym,[-0.1 0.1]); colormap jet;
%         
% X_sort = [];
% for i=1:3
%     X_sort = [ X_sort; data.x_train_encoded( find(grps==i),:)];
% end
% 
% imagesc(X_sort,[-3 3]); colormap redbluecmap;
% 
% subplot(121); gscatter(x(:,1),x(:,2),grps);
% subplot(122); gscatter(x(:,1),x(:,2),data.y_train);

%%
data = load('result_VAE_LINCS_196_organization_d16.mat');
%data = load('result_VAE_LINCS_196_organization_d16.mat');
%data = load('result_VAE_LINCS_196_organization.mat');
%data = load('result_VAE_LINCS_196_organization_DAPI.mat');
%x = fast_tsne(data.x_train_encoded, 3, [], 20, 0.1);
%x = fast_tsne(data.x_train_encoded, 3, [], 30, 0.5);

x = fast_tsne(data.x_train_encoded, 3, [], 20, 0.7);

%load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)

x = bsxfun(@minus, x, min(x)); x = bsxfun(@rdivide, x, max(x));

figure
label = unique(data.y_train);
color_legend = {'r.','g.','b.'};
for i=1:length(label)
    id = []; id = find(data.y_train == label(i));
    scatter3(x(id,1), x(id,2),x(id,3), 200, color_legend{i}); hold on;
end
legend('partially organized', 'unorganized', 'well organized');

figure;
gscatter(x(:,1),x(:,2), data.y_train)
legend('partially organized', 'unorganized', 'well organized');
view(90,90);

tsne1 = x(:,1);
tsne2 = x(:,2);
tsne3 = x(:,3);
latent = data.x_train_encoded;
T = table(tsne1, tsne2, latent, ID, fs);

writetable(T, 'tsne_ID.csv', 'WriteRowNames', true);

%%
rng('default');
opts = statset('Display','final');
N_clust = 3;
Gr = kmeans(zscore(data.x_train_encoded),N_clust,'Replicates',10,'Options',opts);

N = [];
for i=1:N_clust
    id = []; id = find(Gr == i);
    [n,y]=hist(data.y_train(id),[0:2]);
   % n = n/sum(n);
    N = [N; n];
end
bar(N,'stacked'); 
legend('partially organized', 'unorganized', 'well organized');
xticks([1 2 3]); 
xticklabels({'clust 1', 'clust 2', 'clust 3'});
%%
rng('default');
N_clust = 3;

X = (data.x_train_encoded);
Gr = kmeans(X,N_clust,'Replicates',20,'Options',opts);


X_sort = [];
for i=1:3 %N_clust
    ii = []; ii = find(data.y_train == label(i));
    X_sort = [X_sort; X(ii,:)];
end

% for i=1:N_clust
%     ii = []; ii = find(Gr == i);
%     X_sort = [X_sort; X(ii,:)];
% end

figure
id_list = [6 3 7 9 13 2 16 5 11 14 4 8 12 15 1 10];
%imagesc([X(:,id_list) X_sort(:,id_list)],[-4 4]); colormap redbluecmap;
imagesc([X_sort(:,id_list)],[-3 3]); colormap redbluecmap;
%%
close all

[coeff, score, latent, tsquared] = pca(data.x_train_encoded);
%biplot(coeff(:,1:2), 'scores', score(:,1:2));
gscatter(score(:,1),score(:,2), data.y_train);
legend('partially organized', 'unorganized', 'well organized');
hold on;
marker = {'r','g','b'}
for i=1:3
    ij = []; ij = find(data.y_train == label(i));
   
 %   dscatter_new(score(ij,1),score(ij,2), 'plottype','contour')
end

figure
scatterhist(score(:,1),score(:,2), 'Group', data.y_train, 'Kernel','on', ...
            'Location', 'SouthEast','Direction','out');
legend('partially organized', 'unorganized', 'well organized','all');
xlabel('PC #1');
ylabel('PC #2');
%%
X = data.x_train_encoded; %score*coeff';
cg = clustergram(data.x_train_encoded,'Colormap',redbluecmap);
figure
nstep=0.4;
nbin = [-5:nstep:5];
for i=1:size(X,2)
    subplot(4,4,i)
    xx = []; nn = [];
    for j=1:3
        n = []; y = [];
        [n,y] = hist(X(data.y_train == label(j),i), nbin);
        %[n,x] = hist(X(Gr == j,i), nbin);
        n = n/sum(n);
%        bar(x,n,'LineWidth',2.5); hold on;
        nn = [nn; n];
        xx = [xx; y];
    end
    [n_all,x_all] = hist(X(:,i), nbin); 
    n_all = n_all/sum(n_all);
    %nn = [nn; n_all]; xx = [xx; x_all];
    
    plot(xx',nn', 'LineWidth',2.0); %stairs(xx'-nstep,nn', 'LineWidth',2.0)
    hold on;
    plot(x_all, n_all,'k', 'LineWidth',2.0); %stairs(x_all-nstep, n_all,'k', 'LineWidth',2.0);
   % set(gca, 'XScale' ,'log');
   % axis([-5 5 0 80]);
     
 %   bar(x_all, n_all, 'k');
    
    if i== 1
        legend('partially organized', 'unorganized', 'well organized','all');
    end
    ylabel('bin');
    title(sprintf('latent:%d',i));
    grid on;
end

%%
% ii = 0;
% for i=1:16
%     for j=(i+1):16
%         ii = ii + 1;
%         subplot(10,12,ii)
figure;
i = 2; j=12;
scatterhist(data.x_train_encoded(:,i), data.x_train_encoded(:,j), 'Group', data.y_train,'Kernel','on', ...
            'Location', 'SouthEast','Direction','out');
 legend('partially organized', 'unorganized', 'well organized','all');
xlabel(sprintf('latent %d',i));
ylabel(sprintf('latent %d',j));
%%
rng('default');
N_clust = 3;

X = (data.x_train_encoded);


Gr = kmeans(X,N_clust,'Replicates',20,'Options',opts);

%Gr = kmeans(zscore(data.x_train_encoded),N_clust);

N = [];
for i=1:3
    id = []; id = find(data.y_train == label(i));
    [n,y]=hist(Gr(id),[1:N_clust]);
    n = n/sum(n);
    N = [N; n];
end
figure
bar(N,'stacked'); 
xticks([1 2 3]); 
xticklabels({'partially organized', 'unorganized', 'well organized'});
%%

%% create an embedding image

 % size of full embedding image
S = 10000; %size(fs,1)
G = zeros(S, S, 3, 'uint8');
s = 200; % size of every single image
N = length(fs);


Ntake = N;
for i=1:Ntake
    
    if mod(i, 10)==0
        fprintf('%d/%d...\n', i, Ntake);
    end
    
    % location
    a = ceil(x(i, 1) * (S-s)+1);
    b = ceil(x(i, 2) * (S-s)+1);
    a = a-mod(a-1,s)+1;
    b = b-mod(b-1,s)+1;
    if G(a,b,1) ~= 0
        continue % spot already filled
    end
    
    I = imread(fs{i});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    
    mask  = ones(size(I,1),size(I,2));
    mask = mask - bwperim(mask);
    
    switch(data.y_train(i))
        case 0
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [1 0 0]);
        case 1
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [0 1 0]);
            %Ioverlay = imoverlay(I, bwperim(imerode(mask, strel('disk',5))), [0 1 0]);
        case 2
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [0 0 1]);
            %Ioverlay = imoverlay(I, bwperim(imerode(mask, strel('disk',5))), [0 0 1]);
    end
    
    %G(a:a+s-1, b:b+s-1, :) = I;
    G(a:a+s-1, b:b+s-1, :) = Ioverlay;
    
end

figure
imshow(G);

%
imwrite(G, 'VAE_organization_label.png', 'png');

%% average up images
% % (doesnt look very good, failed experiment...)
% 
% S = 1000;
% G = zeros(S, S, 3);
% C = zeros(S, S, 3);
% s = 50;
% 
% Ntake = 5000;
% for i=1:Ntake
%     
%     if mod(i, 100)==0
%         fprintf('%d/%d...\n', i, Ntake);
%     end
%     
%     % location
%     a = ceil(x(i, 1) * (S-s-1)+1);
%     b = ceil(x(i, 2) * (S-s-1)+1);
%     a = a-mod(a-1,s)+1;
%     b = b-mod(b-1,s)+1;
%     
%     I = imread(fs{i});
%     if size(I,3)==1, I = cat(3,I,I,I); end
%     I = imresize(I, [s, s]);
%     
%     G(a:a+s-1, b:b+s-1, :) = G(a:a+s-1, b:b+s-1, :) + double(I);
%     C(a:a+s-1, b:b+s-1, :) = C(a:a+s-1, b:b+s-1, :) + 1;
%     
% end
% 
% G(C>0) = G(C>0) ./ C(C>0);
% G = uint8(G);
% imshow(G);

%% do a guaranteed quade grid layout by taking nearest neighbor

S = 3380; % size of final image
G = zeros(S, S, 3, 'uint8');
s = 200; % size of every image thumbnail


xnum = S/s;
ynum = S/s;
used = false(N, 1);

qq=length(1:s:S);
abes = zeros(qq*2,2);
i=1;
for a=1:s:S
    for b=1:s:S
        abes(i,:) = [a,b];
        i=i+1;
    end
end
%abes = abes(randperm(size(abes,1)),:); % randperm

for i=1:size(abes,1)
    a = abes(i,1);
    b = abes(i,2);
    %xf = ((a-1)/S - 0.5)/2 + 0.5; % zooming into middle a bit
    %yf = ((b-1)/S - 0.5)/2 + 0.5;
    xf = (a-1)/S;
    yf = (b-1)/S;
    dd = sum(bsxfun(@minus, x(:,1:2), [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
    I = imread(fs{di});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);
    
    mask  = ones(size(I,1),size(I,2));
    mask = mask - bwperim(mask);
    
    switch(data.y_train(di))
        case 0
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [1 0 0]);
        case 1
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [0 1 0]);
            %Ioverlay = imoverlay(I, bwperim(imerode(mask, strel('disk',5))), [0 1 0]);
        case 2
            Ioverlay = imoverlay(I, (imerode(mask, strel('disk',5))-imerode(mask, strel('disk',20))), [0 0 1]);
            %Ioverlay = imoverlay(I, bwperim(imerode(mask, strel('disk',5))), [0 0 1]);
    end
    

    G(a:a+s-1, b:b+s-1, :) = Ioverlay;

    if mod(i,10)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end

figure
imshow(G);

%
imwrite(G, 'VAE_organization_label_box.png', 'png');
