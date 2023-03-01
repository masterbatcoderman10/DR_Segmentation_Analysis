clc, clearvars, close all;

%%%%%%%% Testing image enhancement

data_dir = "../../../Data/retinal-lesions-v20191227/fundus_images/";
test_img_path = "9_left.jpg";

test_img_pre = imread(data_dir + test_img_path);
%imshow(test_img);

test_img = medfilt3(test_img_pre, [3,3,3]);

[R, G, B] = imsplit(test_img);

R = adapthisteq(R, "ClipLimit", 0.006);
G = adapthisteq(G, "ClipLimit", 0.006);
B = adapthisteq(B, "ClipLimit", 0.006);

test_img = cat(3, R, G, B);
% test_img = rgb2lab(test_img);
% 
% L = test_img(:,:, 1);
% A = test_img(:,:, 2);
% B = test_img(:,:, 3);
% 
% L = adapthisteq(L, "ClipLimit", 0.006);
% L = im2double(L);
% L = (L - min(L(:))) / (max(L(:)) - min(L(:))); % Rescale the L channel to the range [0, 1]
% L = L * 100; % Scale the L channel back to the range [0, 100]
% 
% 
% test_img = cat(3, L, A, B);
% test_img = lab2rgb(test_img);

figure
subplot(1,2,1)
imshow(test_img_pre)

subplot(1,2,2)
imshow(test_img)


%%%%%%% Image enhancement method 1 & 2

target_dir = "../../../Data/retinal-lesions-v20191227/noise_removed/";
pre_image_files = dir(fullfile(data_dir, "*.jpg"));
image_file_names = {pre_image_files.name};



% for i = 1:length(pre_image_files)
%     
%     img_file = image_file_names{i};
%     img = imread(fullfile(data_dir,img_file));
%     img = medfilt3(img, [3,3,3]);
% 
% %     [R,G,B] = imsplit(img);
% 
%     %R = adapthisteq(R, "ClipLimit", 0.006);
%     %G = adapthisteq(G, "ClipLimit", 0.006);
%     %B = adapthisteq(B, "ClipLimit", 0.006);
% 
%     %enh_img = cat(3, R, G, B);
% 
%     dest_path = target_dir + img_file;
%     %Change to enh_img when applying any enhancement, just img for noise
%     %removal
%     imwrite(img, dest_path);
%     fprintf("Processed image %d\n", i);
% 
% end



%%%%% Image Enhancement 3


% for i = 1:length(pre_image_files)
%     
%     img_file = image_file_names{i};
%     img = imread(fullfile(data_dir,img_file));
% 
%     %Apply median filtering to the image
%     img = medfilt3(img, [3,3,3]);
%     
%     %Convert to LAB and extract channels
%     img = rgb2lab(img);
%     L = img(:,:,1);
%     A = img(:,:,2);
%     B = img(:,:,3);
%     
%     %Enhance only the L channel for lumosity
%     L = adapthisteq(L, "ClipLimit", 0.006);
%     
%     %Merge all the channels
%     enh_img = cat(3, L, A, B);
%     enh_img = lab2rgb(enh_img);
% 
%     dest_path = target_dir + img_file;
% 
%     imwrite(enh_img, dest_path);
%     fprintf("Processed image %d\n", i);
% 
% end



