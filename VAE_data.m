clear all;
clc;
close all;



fname = dir('./img/*.tif');
n_ch = 4;

for i=1:length(fname)
    I = [];
    for j=1:n_ch
        I{j} = double(imread( sprintf('./img/%s', fname(i).name), j));
    end
    
    scale = 255;
    merged = [];
    merged(:,:,3) = adapthisteq(uint8(I{1}/max(I{1}(:))*scale));
    merged(:,:,2) = adapthisteq(uint8(I{2}/max(I{2}(:))*scale));
    %merged(:,:,1) = (uint8((I{3})/max(I{3}(:))*scale));
    merged(:,:,1) = adapthisteq(uint8((I{3}+I{4})/max(I{3}(:)+I{4}(:))*scale));
    
   
    
    merged = imresize(uint8(merged),[256 256]);
    
    imwrite(merged, sprintf('./out/%s.png', fname(i).name(1:end-4)), 'png');
end