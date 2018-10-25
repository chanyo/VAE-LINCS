clear all;
clc;
close all;


dir_train = './train/';
dir_test = './test/';

list_org = dir(sprintf('%s*', dir_train));
  
list = list_org(4:end);


% training   
x_train = []; k = 0;

fileID_train = fopen('list_train_196_organization.txt','w');
for i=1:length(list)
    fname = [];
    fname = dir(sprintf('%s%s/*.png', dir_train, list(i).name));
    
    fprintf('i= %d/%d\n', i, length(list));
    
    
    rand_id = [];
    rand_id = randperm(length(fname));
    for z=1:length(rand_id)
        
        j = rand_id(z);
        
        I = []; I = imread( sprintf('%s%s/%s',dir_train, list(i).name, fname(j).name));
        
        I = imresize(I, [196 196]);
        if i==1 & z==1
            [rows,cols,chns] = size(I);
        end
        I_r = [reshape(I(:,:,1), [1 rows cols]); ...
                reshape(I(:,:,2), [1 rows cols]); ...
                reshape(I(:,:,3), [1 rows cols])]; 
               % reshape(imadd(I(:,:,1), I(:,:,2)), [1 rows cols]); ...
               % reshape(imadd(I(:,:,1), I(:,:,3)), [1 rows cols]); ...
               % reshape(imadd(I(:,:,2), I(:,:,3)), [1 rows cols])];
        k = k + 1;
        x_train(k,:,:,:) = I_r;
        y_train(k,1) = i-1;
        
        fprintf(fileID_train,'%s%s/%s\n',dir_train, list(i).name, fname(j).name);

       
    end
end
fclose(fileID_train);

% 
% 
% % test
% x_test = []; k = 0;
% for i=1:length(list)
%     fname = [];
%     fname = dir(sprintf('%s%s/*.png', dir_test, list{i}));
%     
%     fprintf('i= %d/%d\n', i, length(list));
%     
%     for j=1:length(fname)
%         I = []; I = imread( sprintf('%s%s/%s',dir_test, list{i}, fname(j).name));
%         
%          
%          I_r = [reshape(I(:,:,1), [1 rows cols]); ...
%                 reshape(I(:,:,2), [1 rows cols]); ...
%                 reshape(I(:,:,3), [1 rows cols])];
%         k = k + 1;
%         x_test(k,:,:,:) = I_r;
%         y_test(k,1) = i-1;
%     end
% end


x_test = x_train;
y_test = y_train;
save('LINCS_mat_196_organization.mat', 'x_train','y_train','x_test','y_test','-v7.3');