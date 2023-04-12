from sklearn.metrics import precision_recall_fscore_support, accuracy_score

class ClassificationEvalPipeline:

    def __init__(self, model_dict, dataloader, class_dict, num_classes) -> None:
        
        self.model_dict = model_dict
        _, self.targets = zip(*[(data[0].numpy(), data[1].numpy()) for data in dataloader])
        self.targets = np.argmax(self.targets, axis=-1)
        self.class_dict = class_dict
        self.num_classes = num_classes
    
    def prediction_gen(self):

        predictions = {}
        
        for m in self.model_dict.keys():
            model = self.model_dict[m]

            #For the methods below, the predictions need to be in the following shape : (B,)
            pred = model.predict(self.dl)
            pred = np.argmax(pred, axis=-1)            
            predictions[m] = pred
            
        self.pred = predictions
    
    def stage_one(self, model_keys, path="./stage_one.csv"):
        
        """This function computes standard classification metrics averaged across all classes"""
        scores = {}

        for model_name in model_keys:
            pred = self.pred[model_name]
            acc = accuracy_score(self.gt, pred)
            prec, rec, f1, _ = precision_recall_fscore_support(self.gt, pred, average="macro")

            score = [acc, prec, rec, f1]
            scores[model_name] = score



        