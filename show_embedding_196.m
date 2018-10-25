
close all
clc
clear all
%% load embedding

datatable = readtable('HMEC_240L_COL1.csv');

%% load validation image filenames

fs = textread('list_train_200.txt', '%s');

ID = [];
for j=1:length(fs)
    id = []; id = strfind(fs{j}, '/');
    id_image = []; id_image = str2num(fs{j}(id(end)+1:end-4));
    
    ID = [ID; find(datatable.ImageID == id_image)];
end

% fs = textread('list_train_196.txt', '%s');

N = length(fs);


%%


data = load('result_VAE_LINCS_196.mat');
x = fast_tsne(data.x_train_encoded, 2, 10, 20, 0.7);

%load('imagenet_val_embed.mat'); % load x (the embedding 2d locations from tsne)
x = bsxfun(@minus, x, min(x));
x = bsxfun(@rdivide, x, max(x));

figure
gscatter(x(:,1),x(:,2), data.y_train)


tsne1 = x(:,1);
tsne2 = x(:,2);
latent = data.x_train_encoded;
T = table(tsne1, tsne2, latent, ID);

writetable(T, 'tsne_ID.csv', 'WriteRowNames', true);
%%
figure
N_clust = 5;
%Label_info = []; L_id = []; L_sort = []; mean_L = []; b = []; idx = [];
Label_info = datatable.Cells_PA_Gated_EdUKRT5Class_SE(ID);
%Label_info = datatable.Cells_PA_Gated_EdUKRT19Class_SE(ID);


%Label_info = datatable.Cells_CP_Intensity_IntegratedIntensity_KRT5(ID);
%Label_info = datatable.Cells_CP_Intensity_IntegratedIntensity_KRT19(ID);

%Label_info = datatable.Cytoplasm_CP_Intensity_IntegratedIntensity_KRT5(ID);
%Label_info = datatable.Cytoplasm_CP_Intensity_IntegratedIntensity_KRT19(ID);


%Label_info = datatable.Cytoplasm_CP_AreaShape_Area(ID);

%Label_info = datatable.Cells_CP_Intensity_IntegratedIntensity_KRT19(ID);
Label_info = datatable.Cells_CP_Intensity_IntegratedIntensity_KRT5(ID);

%Label_info = datatable.Cells_CP_Intensity_MedianIntensity_KRT19_SE(ID);
%Label_info = datatable.Cells_CP_Intensity_MedianIntensity_KRT5_SE(ID);

%Label_info = datatable.Cytoplasm_CP_Intensity_IntegratedIntensity_KRT19_SE(ID);
%Label_info = datatable.Cytoplasm_CP_Intensity_IntegratedIntensity_KRT5_SE(ID);

L_id = kmeans((Label_info), N_clust, 'Replicates', 10, 'MaxIter', 50);
for j=1:N_clust
    mean_L(j) = mean( Label_info(find(L_id == j)));
end

[b, idx] = sort(mean_L);
for j=1:N_clust
    L_sort( find(L_id == idx(j))) = j;
end
gscatter(x(:,1),x(:,2), L_sort)
view(90,90);
%%
figure
N_clust = 8;
L = kmeans( zscore(data.x_train_encoded),N_clust);
gscatter(x(:,1),x(:,2), L);%data.y_train)

class_label = unique(data.y_train);
% 
% N = [];
% for i=1:length(class_label)
%     id = []; id = find(data.y_train == class_label(i));
%     
%     [n,~] = hist(L(id), 1:N_clust);
%     N = [N; n/sum(n)];
% end






%% create an embedding image

S = 11400; % size of full embedding image
S = 15000; 
G = zeros(S, S, 3, 'uint8');
s = 196; % size of every single image

Ntake = N;
for i=1:Ntake
    
    if mod(i, 100)==0
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
    
    G(a:a+s-1, b:b+s-1, :) = I;
    
end

figure
imshow(G);

%%
imwrite(G, 'cnn_embed_2k_196.png', 'png');

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

S = 11400; % size of final image
G = zeros(S, S, 3, 'uint8');
s = 196; % size of every image thumbnail


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
    dd = sum(bsxfun(@minus, x, [xf, yf]).^2,2);
    dd(used) = inf; % dont pick these
    [dv,di] = min(dd); % find nearest image

    used(di) = true; % mark as done
    I = imread(fs{di});
    if size(I,3)==1, I = cat(3,I,I,I); end
    I = imresize(I, [s, s]);

    G(a:a+s-1, b:b+s-1, :) = I;

    if mod(i,100)==0
        fprintf('%d/%d\n', i, size(abes,1));
    end
end

figure
imshow(G);

%%
imwrite(G, 'cnn_embed_full_2k_196.png', 'png');
