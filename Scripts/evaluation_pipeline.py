import json
class EvalPipeline:
    
    def __init__(self, gt, img, n,model_dict, class_dict):
        
        self.gt = gt
        self.img = img
        self.model_dict = model_dict
        self.n = n
        self.class_dict = class_dict

        self.prediction_gen()
        
    def prediction_gen(self):
        
        predictions = {}
        
        for m in self.model_dict.keys():
            model = self.model_dict[m]
   
            pred = model.predict(self.img)
            pred = np.array(pred)
            pred = np.argmax(pred, axis=3)
            pred = pred.reshape(pred.shape[0], pred.shape[1], pred.shape[2], 1)
            
            predictions[m] = pred
            
        self.pred = predictions
            
    
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
        
        #Assumes the images aren't 4 dimensional tensors
        
        #Number of images should be the same as G.T
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
            image = load_img(img_path)
            ax[0, i].imshow(image)
            
            ax[0,i].set_xticks([])
            ax[0, i].set_yticks([])
        
        #Plotting the ground truth
        for i, ann in enumerate(gt_files):
            
            img_path = os.path.join(gt_dir, ann)
            image = tf.io.read_file(img_path)
            image = tf.io.decode_png(image)

            ax[1, i].imshow(image)
            
            ax[1,i].set_xticks([])
            ax[1, i].set_yticks([])
        
        for i, model_name in enumerate(self.model_dict.keys()):
            
            i = i+2
            model = self.model_dict[model_name]
            ax[i, 0].set_ylabel(model_name)
            
            for n, img in enumerate(img_files):
                
                img_path = os.path.join(img_dir, img)
                image = tf.io.read_file(img_path)
                image = tf.io.decode_jpeg(image, 3)
                image = tf.image.resize(image, [224, 224])
                image = np.expand_dims(image, axis=0)
                pred = model.predict(image)
                pred = np.squeeze(pred)
                
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
        
        
        
    