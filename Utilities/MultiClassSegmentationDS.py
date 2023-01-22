class MultiClassSegmentationDataset(Dataset):
    
    def __init__(self, image_files, mask_dirs, size, classes,n_classes):
        
        assert len(image_files) == len(mask_dirs)
        
        self.image_files = sorted(image_files)
        self.mask_dirs = sorted(mask_dirs)
        self.annotation_files = {}
        
        for img_fldr in self.mask_dirs:
            self.annotation_files[img_fldr] = sorted(os.listdir(img_fldr))
            
        self.H = size
        self.W = size
        self.n = n_classes
        self.classes = dict(zip(sorted(classes), range(n_classes)))

    
    def __len__(self):
        
        return len(self.image_files)

    def __getitem__(self, idx):
        
        #Reading the image
        img_path = self.image_files[idx]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.H,self.W))
        #norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = img / 255.0
        norm_img = np.transpose(norm_img, (2, 0, 1))
        img_tensor = torch.from_numpy(norm_img)
        img_tensor = img_tensor.float()
        
        
        #Preparing mask - There are 4 lesion classes, the directories have 1 picture each for a specific lesion, have to combine them
        #1 extra class for the background
        #The background mask is currently all ones, the ones from each class have to be subtracted from it
        background = np.ones((self.H, self.W))
        mask = np.zeros((self.n + 1, self.H, self.W))
        ann_fldr = self.mask_dirs[idx]
        ann_files = self.annotation_files[ann_fldr]
        for ann_file in ann_files:
            
            code = ann_file[-6:-4]
            ix = self.classes[code]
            ann_path = os.path.join(ann_fldr, ann_file)
            ann = cv2.imread(ann_path, 0)
            
            #update background by subtracting the current pixel positions
            background = background - ann
            
            ann = np.expand_dims(ann, axis=0)
            
            mask[ix+1, :, :] = ann
        
        #If there are overlapping pixel values, the pixel value at a certain position may be negative, have to clip to 0
        background = np.clip(background, 0, None)
        background = np.expand_dims(background, axis=0)
        #Place the background class inside
        mask[0,:,:] = background
        
        mask_tensor = torch.from_numpy(mask)
        mask_tensor = mask_tensor.float()
        
        return img_tensor, mask_tensor
        
        
                
        
        
        
        