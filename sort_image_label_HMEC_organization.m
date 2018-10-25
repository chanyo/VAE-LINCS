clear all;
clc;
close all


fname = dir('./image-org/*.png');

meta = readtable('list_image_selected.csv');



%D = readtable('HMEC_240L_COL1.csv');
D = readtable('all_meta.csv');

Unique_label = unique(meta.PopShape);

ID = []; Rate = [];
for i=1:length(fname)
    id = find(D.ImageID == str2num(fname(i).name(1:end-4)));
    ID = [ID; D.ClarionID(id)];
    
    ii = find(meta.ImageID == D.ClarionID(id)); label = [];
    for j=1:length(ii)
        label(j) = find(strcmp(Unique_label, meta.PopShape(ii(j)))==1);
    end
    Rate = [Rate; mode(label)];
end

%%
organization = unique(Rate);

for j=1:length(organization)
    mkdir (sprintf('./train/%s', num2str(organization(j))));
end


for i=1:length(fname)
     
    movefile( sprintf('./image-org/%s', fname(i).name), ...
                sprintf('./train/%s/%s.png', num2str(Rate(i)), num2str(ID(i))));
    
    
end