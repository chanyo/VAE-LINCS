clear all;
clc;
close all;

%D = readtable('HMEC_240L_COL1.csv');
D = readtable('all_meta.csv');

S = readtable('list_image_single.csv');

%%

ImageID = [];
for i=1:size(S,1)
    id = find(D.ClarionID == S.ImageID(i));
    if isempty(id) == 0
        ImageID = [ImageID; D.ImageID(id)];
    end
end