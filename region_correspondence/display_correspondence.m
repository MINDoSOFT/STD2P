clear
clc

videoID = 1;
r = 425;
c = 560;

csp = rand(10000,3)*255;  % color panel
csp(1,:) = 0;

files = dir(sprintf('./correspondences/%04d',videoID));

frameStep = 3;
startFrame = 1;
targetFrame = 151;

figure;

for i = 3:length(files)
    name = files(i).name;
    corr = load(sprintf('./correspondences/%04d/%s',videoID,name));
    corr = reshape(corr,[r,c]);
    imshow(ind2rgb(corr, im2double(uint8(csp))));
    if ((((i - 3) * frameStep) + startFrame) == targetFrame)
      title(sprintf('Target Frame %05d', ((i - 3) * frameStep) + startFrame));  
    else
      title(sprintf('Frame %05d', ((i - 3) * frameStep) + startFrame));
    end
    drawnow;
end