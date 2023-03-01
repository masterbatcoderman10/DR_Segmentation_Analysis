class ClassificationDataset(Dataset):

    def __init__(self, image_files, target_dir, size, n_classes, test_mode=False):

        self.image_files = image_files
        self.target_dir = target_dir
        self.size = size
        self.n_classes = n_classes
        self.test_mode = test_mode
    
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

        #Creating the target class vector
        target = torch.zeros(self.n_classes)
        #The name of the image file is the last part of the img path, without the extension
        img_file = img_path.split('/')[-1][:-4] + ".txt"
        target_file_path = os.path.join(self.target_dir, img_file)
        with open(target_file_path, "r") as target_file:
            class_ix = int(target_file.read())
        target[class_ix] = 1.0

        if not self.test_mode:
            #Data augmentation if it's not test_mode, for example training and validation data will get augmented
            augment_mode = np.random.randint(0,4)
            if augment_mode == 0:
                img_tensor = F.hflip(img_tensor)
            elif augment_mode == 1:
                img_tensor = F.vflip(img_tensor)
            elif augment_mode == 2:
                angle = np.random.randint(-5, 5)
                img_tensor = F.rotate(img_tensor, angle)


        return img_tensor, target

