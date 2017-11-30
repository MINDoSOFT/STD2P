% run()
%
% Compute region correspondence with superpixel and optical flow under
% intersection over union criterion.
%
% The code establish region correspondence in NYUDv2 dataset (http://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
% We assume you already have the superpixel for each frame and the forward and backward optical
% flow in each adjacent frames.
%
% If you use this code, please cite our paper:
%
% Yang He, Wei-Chen Chiu, Margret Keuper, Mario Fritz, STD2P: RGBD Semantic
% Segmentation using Spatio-Temporal Data-Driven Pooling, CVPR, 2017.
%
% Copyright by Yang He, Wei-Chen Chiu, Margret Keuper and Mario Fritz, 2017

% Change dir into correct folder
strSplit = strsplit(pwd, filesep);
if ( strcmp(strSplit(end), 'region_correspondence')~= 1 )
  cd(['..' filesep 'region_correspondence'])
end

addpath(['.' filesep 'include']);

% load('../config/nyud2_split.mat');
% videoID = trainval; % compute the region correspondence for training set
if (exist('multipleTargets') == 1)
  if (exist('currentTarget') == 1)
    countTargets = multipleTargets;
    videoID = [currentTarget];
  else
    disp('You need to set currentTarget to run this script in multipleTargets mode.');
    return;
  end
else 
  videoID = [1];
end

%disp(videoID);

%return;

% input path
for_std2p_path = ['..' filesep '..' filesep '..' filesep 'data' filesep 'for_std2p'];
%for_std2p_path = 'C:\Users\MiNdOs\Documents\MATLAB\Thesis\datasets\outputs\for_std2p';
img_path = sprintf('%s', for_std2p_path);
superpixel_path = sprintf('%s', for_std2p_path);
superpixel_subdir = 'superpixel';
superpixel_var_name = 'sp';
flow_path = sprintf('%s', for_std2p_path);
flow_subdir = 'flow';
% output path
output_path = sprintf('%s', for_std2p_path);
output_subdir = 'correspondences';

imgColorSuffix = '_color.png';
flowFileSuffix = '.flo';
superpixelFileSuffix = '.mat';
correspondenceSuffix = '.txt';

% parameter for computing region correspondence
interval = 3;        % frame step
nBuf = 3;            % buffer size
threshold = 0.4;     % intersection over union threshold to establish correspondence
display = 1;         % 1 for displaying visualization result during computation, and 0 is not.
% fast mode is useful for very large images which has too much superpixels
fast    = 0;         % 1 for fast computation
thre_sp = 1000;      % threshold for fast computation, increase this parameter for faster computation but less correspondence

% load video information
csp = rand(10000,3)*255;  % color panel
csp(1,:) = 0;

order = load(['..' filesep 'config' filesep 'my_order.txt']);

if (exist('multipleTargets') == 1)
  order = reshape(order,[2,countTargets])';
else
  order = reshape(order,[2,1])';
end
folders = {};
targets = [];
fd = fopen(['..' filesep 'config' filesep 'my_folder.txt']);
while ~feof(fd)
    folders = [folders fgetl(fd)];
end
fclose(fd);
fd = fopen(['..' filesep 'config' filesep 'my_target.txt']);
while ~feof(fd)
    targets = [targets str2num(fgetl(fd))];
end
fclose(fd);

% start to run
for i = videoID

    target = targets(i);
    folder = folders{i}
    strVideoID = sprintf('%04d', i);
    strTarget = sprintf('%05d', target);
    
    if ~exist([output_path filesep folder filesep strVideoID])
        mkdir([output_path filesep folder filesep strVideoID]);
    end
    
    if ~exist([output_path filesep folder filesep strVideoID filesep output_subdir])
        mkdir([output_path filesep folder filesep strVideoID filesep output_subdir]);
    end

    flag = 0;
    output1 = [];
    Buf = [];
    
    img_target = imread([img_path filesep folder filesep strTarget imgColorSuffix]);
    
    for fi = target:3:order(i,2)-3
        disp(sprintf('%d : %d',i, fi));
        flag1 = 1;
        strFi = sprintf('%05d', fi);
        
        img_current = imread([img_path filesep folder filesep strFi imgColorSuffix]);
        
        corr = readFlowFile([flow_path filesep folder filesep flow_subdir filesep sprintf('%05d_%05d', fi, fi+3) flowFileSuffix]);
        vx = corr(:,:,1); % STD2P has 46:470 x 41:600
        vy = corr(:,:,2);
        
        tmp = getfield(load([superpixel_path filesep folder filesep superpixel_subdir filesep sprintf('%05d', fi) superpixelFileSuffix], superpixel_var_name), superpixel_var_name );
        tmp = tmp + 1;
        if fast && fi ~= target
            unique_tmp = unique(tmp);
            for iter1 = 1:length(unique_tmp)
                pos = find(unique_tmp(iter1) == tmp);
                if length(pos) < thre_sp
                    tmp(pos) = -1;
                end
            end
        end
        % corr1 = reshape(tmp,[425,560]);
        corr1 = tmp;
        % disp('Corr1 size:');
        % disp(size(corr1));

        tmp = getfield(load([superpixel_path filesep folder filesep superpixel_subdir filesep sprintf('%05d', fi+3) superpixelFileSuffix], superpixel_var_name), superpixel_var_name );
        tmp = tmp + 1;
        if fast
            unique_tmp = unique(tmp);
            for iter1 = 1:length(unique_tmp)
                pos = find(unique_tmp(iter1) == tmp);
                if length(pos) < thre_sp
                    tmp(pos) = -1;
                end
            end
        end
        % corr2 = reshape(tmp,[425,560]);
        corr2 = tmp;
        % disp('Corr2 size:');
        % disp(size(corr2));
       
        if fi == target          % init
            corr_target_original = corr1;
            corr_target_current = corr1;
            Buf = corr1;
            Buf2 = corr1;
            number = length(unique(corr1(:)));
            set_whole = cell(1,number);
            current_label = unique(corr1(:));
        end
        
        if size(Buf,3) > nBuf    % replace early frame.
           Buf = Buf(:,:,2:nBuf+1);
           Buf2 = Buf2(:,:,2:nBuf+1);
        end
        
        seg_buf = unique(Buf(:));
        Buf = warpImage(Buf,round(-vx),round(-vy));
        segs = unique(corr2(:));
        corr_flow2 = corr2;
        
        tmpBuf = zeros(size(Buf2));
        tmp = corr_flow2;
        for kk = 1:size(Buf2,3)
            corr = readFlowFile([flow_path filesep folder filesep flow_subdir filesep sprintf('%05d_%05d', fi+6-kk*3, fi+3-kk*3) flowFileSuffix]);
            corrSize = size(corr);
            
            vx = corr(:, :,1);
            vy = corr(:, :,2);
        
            tmp = warpImage(tmp,round(-vx),round(-vy));
            % size(tmpBuf(:,:,size(Buf2,3)-kk+1))
            % size(tmp)
            tmpBuf(:,:,size(Buf2,3)-kk+1) = tmp;
        end
        
        matching_score = [];
        for ns = 1:size(Buf,3)
            tmp_source = Buf(:,:,ns) + 0.02;
            tmp_target = corr2 - 0.02;
            tmp_iou = tmp_source.*tmp_target;
            info_source = countUnique(tmp_source);
            info_target = countUnique(tmp_target);
            
            num_source = repmat(info_source(:,2),[1,size(info_target,1)]);
            num_target = repmat(info_target(:,2)',[size(info_source,1),1]);
            num_iou = zeros(size(num_target));
            
            segs_source = unique(tmp_source(:));
            
            for iter1 = 1:size(info_source,1)
                for iter2 = 1:size(info_target,1)
                    num_iou(iter1,iter2) = sum(info_source(iter1,1)*info_target(iter2,1) == tmp_iou(:));
                end
            end
            score_graph = num_iou./(num_source+num_target-num_iou);
            score_graph(find(score_graph==Inf)) = 1;
            
            tmp_source2 = Buf2(:,:,ns);
            tmp_target2 = tmpBuf(:,:,ns);
            segs_source2 = unique(tmp_source2(:));
            for iter1 = 1:length(segs_source2)
                if segs_source2(iter1) == -1
                    continue;
                end
                for iter2 = 1:length(segs)
                    ip = find(segs_source2(iter1) == segs_source);
                    if length(ip) == 0 || score_graph(ip,iter2) < threshold
                        continue;
                    end
                    set_source = find(tmp_source2 == segs_source2(iter1));
                    set_current = find(tmp_target2 == segs(iter2));
                    if length(set_source) == 0 || length(set_current) == 0
                        score_graph(ip,iter2) = 0;
                    else
                        score_graph(ip,iter2) = min(score_graph(ip,iter2),length(intersect(set_source,set_current))/length(union(set_source,set_current)));
                    end
                end
            end
            
            score_graph(score_graph<threshold) = 0;

            for iter1 = 1:length(segs)
                if sum(score_graph(:,iter1)) == 0
                    corr_flow2(find(corr2 == segs(iter1))) = -1;
                    continue;
                end
                [v,p] = max(score_graph(:,iter1));
                region_label = segs_source(p);
                region_score = v;
                matching_score = [matching_score; [segs(iter1), region_label, region_score] ];
            end
        end
        corr3 = zeros(size(corr_flow2));
        for iter1 = 1:length(segs)
            if length(matching_score) ==0
                break;
            end
            pos = find(matching_score(:,1) == segs(iter1));
            if length(pos) == 0
                continue;
            end
            tmp = matching_score(pos,2:3);
            corr_flow2(find(corr2 == segs(iter1))) = round(tmp(argmax(tmp(:,2),1),1));
        end        
        if display
            subplot(2,2,1);imshow(img_target); title('Target frame');
            subplot(2,2,2);imshow(img_current); title('Current frame');
            subplot(2,2,3);imshow(ind2rgb(corr_target_current, im2double(uint8(csp)))); title('Superpixel in target frame');
            subplot(2,2,4);imshow(ind2rgb(corr_flow2, im2double(uint8(csp)))); title('Region correspondence');

            drawnow;  
        end

        corr_target = corr_flow2;
        Buf = cat(3,Buf,corr_flow2);
        Buf2 = cat(3,Buf2,corr_flow2);
        output1 = cat(3,output1,corr_target);
    end
    
    if exist('flag1')
        output1 = cat(3,corr_target_current,output1);
    end

    for fi = target:-3:order(i,1)+3
        disp(sprintf('%d : %d',i, fi));
        strFi = sprintf('%05d', fi);
        
        img_current = imread([img_path filesep folder filesep strFi imgColorSuffix]);
               
        corr = readFlowFile([flow_path filesep folder filesep flow_subdir filesep sprintf('%05d_%05d', fi, fi-3) flowFileSuffix]);
        
        vx = corr(:, :,1);
        vy = corr(:, :,2);

        tmp = getfield(load([superpixel_path filesep folder filesep superpixel_subdir filesep sprintf('%05d', fi) superpixelFileSuffix], superpixel_var_name), superpixel_var_name );
        tmp = tmp + 1;
        if fast && fi ~= target
            unique_tmp = unique(tmp);
            for iter1 = 1:length(unique_tmp)
                pos = find(unique_tmp(iter1) == tmp);
                if length(pos) < thre_sp
                    tmp(pos) = -1;
                end
            end
        end
        % corr1 = reshape(tmp,[corrSize(1),corrSize(2)]);
        corr1 = tmp;
        tmp = getfield(load([superpixel_path filesep folder filesep superpixel_subdir filesep sprintf('%05d', fi-3) superpixelFileSuffix], superpixel_var_name), superpixel_var_name );
        tmp = tmp + 1;
        if fast
            unique_tmp = unique(tmp);
            for iter1 = 1:length(unique_tmp)
                pos = find(unique_tmp(iter1) == tmp);
                if length(pos) < thre_sp
                    tmp(pos) = -1;
                end
            end
        end
        % corr2 = reshape(tmp,[corrSize(1),corrSize(2)]);
        corr2 = tmp;
       
        if fi == target          
            corr_target_original = corr1;
            corr_target_current = corr1;
            Buf2 = corr1;
            Buf = corr1;
            number = length(unique(corr1(:)));
            set_whole = cell(1,number);
            if ~exist('current_label')
                current_label = unique(corr1(:));
            end
        end
        
        if size(Buf,3) > nBuf
           Buf = Buf(:,:,2:nBuf+1);
           Buf2 = Buf2(:,:,2:nBuf+1);
        end
        
        seg_buf = unique(Buf(:));
        Buf = warpImage(Buf,round(-vx),round(-vy));
        
        segs = unique(corr2(:));
        corr_flow2 = corr2;
        
        tmpBuf = zeros(size(Buf2));
        tmp = corr_flow2;
        for kk = 1:size(Buf2,3)
            corr = readFlowFile([flow_path filesep folder filesep flow_subdir filesep sprintf('%05d_%05d', fi-6+kk*3, fi-3+kk*3) flowFileSuffix]);
            vx = corr(:, :,1);
            vy = corr(:, :,2);
            [tmp] = warpImage(tmp,round(-vx),round(-vy));
            tmpBuf(:,:,size(Buf2,3)-kk+1) = tmp;
        end
        
        matching_score = [];
        for ns = 1:size(Buf,3)
            tmp_source = Buf(:,:,ns) + 0.02;
            tmp_target = corr2 - 0.02;
            tmp_iou = tmp_source.*tmp_target;
            info_source = countUnique(tmp_source);
            info_target = countUnique(tmp_target);
            
            num_source = repmat(info_source(:,2),[1,size(info_target,1)]);
            num_target = repmat(info_target(:,2)',[size(info_source,1),1]);
            num_iou = zeros(size(num_target));
            
            segs_source = unique(tmp_source(:));
            
            for iter1 = 1:size(info_source,1)
                for iter2 = 1:size(info_target,1)
                    num_iou(iter1,iter2) = sum(info_source(iter1,1)*info_target(iter2,1) == tmp_iou(:));
                end
            end
            score_graph = num_iou./(num_source+num_target-num_iou);
            score_graph(find(score_graph==Inf)) = 1;
            
            tmp_source2 = Buf2(:,:,ns);
            tmp_target2 = tmpBuf(:,:,ns);
            segs_source2 = unique(tmp_source2(:));
            for iter1 = 1:length(segs_source2)
                if segs_source2(iter1) == -1
                    continue;
                end
                for iter2 = 1:length(segs)
                    ip = find(segs_source2(iter1) == segs_source);
                    if length(ip) == 0 || score_graph(ip,iter2) < threshold
                        continue;
                    end
                    set_source = find(tmp_source2 == segs_source2(iter1));
                    set_current = find(tmp_target2 == segs(iter2));
                    if length(set_source) == 0 || length(set_current) == 0
                        score_graph(ip,iter2) = 0;
                    else
                        score_graph(ip,iter2) = min(score_graph(ip,iter2),length(intersect(set_source,set_current))/length(union(set_source,set_current)));
                    end
                end
            end
            
            score_graph(score_graph < threshold) = 0;
            
            for iter1 = 1:length(segs)
                if sum(score_graph(:,iter1)) == 0
                    corr_flow2(find(corr2 == segs(iter1))) = -1;
                    continue;
                end
                [v,p] = max(score_graph(:,iter1));
                region_label = segs_source(p);
                region_score = v;
                matching_score = [matching_score; [segs(iter1), region_label, region_score] ];
            end
        end
        for iter1 = 1:length(segs)
            if length(matching_score) ==0
                break;
            end
            pos = find(matching_score(:,1) == segs(iter1));
            if length(pos) == 0
                continue;
            end
            tmp = matching_score(pos,2:3);
            corr_flow2(find(corr2 == segs(iter1))) = round(tmp(argmax(tmp(:,2),1),1));
        end
        if display
            subplot(2,2,1);imshow(img_target); title('Target frame');
            subplot(2,2,2);imshow(img_current); title('Current frame');
            subplot(2,2,3);imshow(ind2rgb(corr_target_current, im2double(uint8(csp)))); title('Superpixel in target frame');
            subplot(2,2,4);imshow(ind2rgb(corr_flow2, im2double(uint8(csp)))); title('Region correspondence');

            drawnow;  
        end

        if ~exist('flag1') && target == fi
            output1 = cat(3,corr_target_current,output1);
        end
        
        corr_target = corr_flow2;
        Buf = cat(3,Buf,corr_flow2);
        Buf2 = cat(3,Buf2,corr_flow2);
        output1 = cat(3,corr_target,output1);

    end

    % saving the region correspondence
    kk = 1;
    for iter1 = order(i,1):3:order(i,2)
        strIter1 = sprintf('%05d', iter1);
        tmp = output1(:,:,kk);
        tmp = tmp(:);
        outputFilename = [output_path filesep folder filesep strVideoID filesep output_subdir filesep strIter1 correspondenceSuffix];
        disp(outputFilename)
        fp = fopen(outputFilename,'w');
        fprintf(fp,'%g\n',tmp);
        fclose(fp);
        kk = kk + 1;
    end
    close all;
end

disp('All done.')
