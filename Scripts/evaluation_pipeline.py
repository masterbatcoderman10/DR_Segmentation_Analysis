import json

class EvalPipeline:
    
    def __init__(self, dataloader, n,model_dict, class_dict, color_dict):
        
        """"This is an evaluation pipeline which can be used to determine the evaluation metrics for a group of segmentation models."""

        self.dl = dataloader
        self.img, self.gt = zip(*[(data[0].numpy(), data[1].numpy()) for data in dataloader])
        self.img = np.concatenate(self.img)
        #For the methods defined below, the ground-truths need to be in the following shape : (B, H, W, 1)
        self.gt = np.concatenate(self.gt)

        self.gt = np.transpose(self.gt, (0, 2, 3, 1))
        self.gt = np.argmax(self.gt, axis=3)
        self.gt = np.expand_dims(self.gt, axis=-1)
        
        self.model_dict = model_dict
        self.n = n
        self.class_dict = class_dict
        self.color_dict = color_dict
        self.prediction_gen()
        
    def prediction_gen(self):

        """Generates predictions for each segmentation model"""
        
        predictions = {}
        
        for m in self.model_dict.keys():
            model = self.model_dict[m]

            #For the methods below, the predictions need to be in the following shape : (B, H, W, 1)
            pred = model.predict(self.dl)
            pred = np.argmax(pred, axis=-1)
            pred = np.expand_dims(pred, axis=-1)
            
            predictions[m] = pred
            
        self.pred = predictions
    
    #This function swaps colors from a class map
    def color_swap(self, img):
    
        for key in self.color_dict.keys():

            c = np.where(img[:, :, [0,1,2]] == [key, key, key])
            img[c[0], c[1], :] = self.color_dict[key]

        return img
            
    
    def stage_one(self, metrics=["SENS", "SPEC", "IoU", "DSC"], path="stage_1.csv"):
        
        scores = {}
        
        #First evaluate the same metric for all sets of predictions
        print(metrics)
        for metric in metrics:
            
            scores[metric] = []
            for p in self.pred.keys():
                
                
                current_pred = self.pred[p]
                #Evaluate for the current prediction
                score = evaluate(self.gt, current_pred, metric=metric, multi_class=True, n_classes=self.n)
                score = np.mean(score)
                scores[metric].append(score)
                
        with open(path, "w") as f:
            for m in metrics:
                f.write(m)
                f.write(",")
            f.write("\n") 
            for i in range(len(scores[m])):
                for m in metrics:
                
                    f.write(str(scores[m][i]))
                    f.write(",")
                f.write("\n")
            
            
        return scores
    
    def stage_two(self, metrics=["SENS", "SPEC", "IoU", "DSC"], path="stage_2.json"):
        
        scores = {}
        
        # Have to change all of done_pred instances into self.pred
        for p in self.pred.keys():
            scores[p] = []
            print(f"Working on : {p}...")
            for metric in metrics:
                
                current_pred = self.pred[p]
                #Evaluate for the current prediction
                score = evaluate(self.gt, current_pred, metric=metric, multi_class=True, n_classes=self.n)
                scores[p].append(score)
            
            scores[p] = np.array(scores[p]).T
        
        scores_2 = {}
        
        print("Creating final dict")
        for p in self.pred.keys():
            scores_2[p] = {}
            for i, c in enumerate(self.class_dict):
                scores_2[p][c] = {}
                for n, m in enumerate(metrics):
                    scores_2[p][c][m] = scores[p][i][n]
        
        

        with open(path, "w") as json_file:
            json.dump(scores_2, json_file)
        
        
        return scores_2

    
    def stage_four(self, img_dir, gt_dir, img_files, gt_files, path="stage_4.png"):
        
        """This method is used to compare the predictions of the models passed in with the ground truth for all the images passed in as arguments into this function.
        img_dir : this is the directory of the images : string
        gt_dir : this is the directory where the ground truth images are present : string
        img_files : the specific image file names choosen to plot : list of strings
        gt_files : the corresponding ground truth files choosen to comapare against : list of strings
        The img_files and gt_files must be present within the img_dir and gt_dir respectively.
        Furthermore, the same number of image files and ground truth files should be passed in.
        """
        
        #Number of images should be the same as G.T
        assert len(img_files) == len(gt_files)
        print(img_dir)
        print(gt_dir)
        print(img_files)
        print(gt_files)
        
        n_cols = len(img_files)
        n_rows = len(list(self.model_dict.keys())) +2
        
        #Defining the figure
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3,n_rows*3))
        
        ax[0, 0].set_ylabel("images")
        ax[1, 0].set_ylabel("ground_truth")
        
        #Plotting the images first
        for i, img in enumerate(img_files):
            
            img_path = os.path.join(img_dir, img)
            image = cv2.imread(img_path)
            ax[0, i].imshow(image)
            
            ax[0,i].set_xticks([])
            ax[0, i].set_yticks([])
        
        #Plotting the ground truth
        for i, ann in enumerate(gt_files):
            
            gt_path = os.path.join(gt_dir, ann)
            gt = cv2.imread(gt_path)
            gt = cv2.resize(gt, (512, 512))

            ax[1, i].imshow(gt)
            
            ax[1,i].set_xticks([])
            ax[1, i].set_yticks([])
        
        for i, model_name in enumerate(self.model_dict.keys()):
            
            i = i+2
            model = self.model_dict[model_name]
            ax[i, 0].set_ylabel(model_name)
            
            for n, img in enumerate(img_files):
                
                img_path = os.path.join(img_dir, img)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (512, 512))
                image = image / 255.0
                image = np.transpose(image, (2, 0, 1))
                image = torch.from_numpy(image).float()
                
                pred = model.predict_image(image)
                pred = np.argmax(pred, axis=-1)
                pred = np.transpose(pred, (1,2,0))
                pred = np.repeat(pred, repeats=3,axis=-1)
                pred = self.color_swap(pred)
                
                ax[i, n].imshow(pred)
                ax[i,n].set_xticks([])
                ax[i,n].set_yticks([])
            
        plt.subplots_adjust(left=0.2,
                    right=0.9,
                    wspace=0.4,
                    hspace=0.4)
            
        plt.savefig(path, dpi=100)
        
    
    ### Stage 5 : F1 score plots per model per class
    
    def stage_five(self, path="stage_5.png"):
        

        
        markers = ["x", "+", ".", "1", "*", "d"]
        colors = ["lime", "fuchsia", "darkorange", "gold", "salmon", "indigo"]
        
        model_names = list(self.pred.keys())
        
        fig, ax = plt.subplots(1,1, figsize=(6,self.n))
        ys = range(self.n)
        
        for i, model_name in enumerate(model_names):
            
            pred = self.pred[model_name]
            
            prec = evaluate(self.gt, pred, metric="PREC", multi_class=True, n_classes=self.n)
            recall = evaluate(self.gt, pred, metric="Recall", multi_class=True, n_classes=self.n)
            
            f1 = (2 * prec * recall) / (prec + recall)
            
            some_num = np.random.uniform(0.1, 0.3, 1)
            
            ax.scatter(x=f1, y=ys, color=colors[i], marker=markers[i], label=model_name)
            ax.set_yticks(ticks=[0,1,2],labels=self.class_dict)
            ax.legend(loc="best")
            ax.grid()
        
        plt.savefig(path, dpi=100)
        
        
        
    