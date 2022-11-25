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
        
        
        
    