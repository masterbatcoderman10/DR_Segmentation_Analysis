class BinarySegmentationDataset(Dataset):
    
    def __init__(self, images_paths, mask_paths):
        
        self.image_files = images_paths
        self.mask_files = mask_paths
        
    def __len__(self):
        
        return len(self.image_files)

    def __getitem__(self, idx):
        
        #Reading the image
        img_path = self.image_files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        norm_img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_img = norm_img.reshape(norm_img.shape[-1], norm_img.shape[0], norm_img.shape[1])
        img_tensor = torch.from_numpy(norm_img)
        img_tensor = img_tensor.float()
        
        #Reading the mask/annotation
        ann_path = self.mask_files[idx]
        ann = cv2.imread(ann_path)
        ann = cv2.cvtColor(ann, cv2.COLOR_BGR2GRAY)
        ann_tensor = torch.from_numpy(ann)
        ann_tensor = ann_tensor.float()
        
        return img_tensor, ann_tensor
        
    
    