clear all;
clc;
close all


fname = dir('./image/*.png');

meta = readtable('HMEC_240L_COL1.csv');


ligand = unique(meta.Ligand);

for j=1:length(ligand)
    mkdir (sprintf('./train/%s', ligand{j}));
end


for i=1:length(fname)
    id = [];
    id = find(strcmp(ligand ,meta.Ligand(i))==1);
    
    movefile( sprintf('./image/%s', fname(i).name), ...
                sprintf('./train/%s/%s', ligand{id}, fname(i).name));
    
    
end