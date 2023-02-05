class Evaluator:
    
    def __init__(self, eval_pipeline, results_dir):
        
        self.eval_pipeline = eval_pipeline
        self.results_dir = results_dir
    
    def handle_one(self, args):
        
        if len(args) == 2:
            keys, path = args
            path = os.path.join(self.results_dir, path)
            self.eval_pipeline.stage_one(model_keys=keys, path=path)
        else:
            mets, keys, path = args
            path = os.path.join(self.results_dir, path)
            self.eval_pipeline.stage_one(mets, keys, path)
    
    def handle_two_or_five(self, routine, args):
        
        if len(args) == 3:
            keys, path, csv_path = args
            path = os.path.join(self.results_dir, path)
            csv_path = os.path.join(self.results_dir, csv_path)
            if routine == 2:
                self.eval_pipeline.stage_two(model_keys=keys, path=path, csv_path=path)
            else:
                self.eval_pipeline.stage_five(keys, path, csv_path)
        else:
            mets, keys, path, csv_path = args
            path = os.path.join(self.results_dir, path)
            csv_path = os.path.join(self.results_dir, csv_path)
            self.eval_pipeline.stage_two(mets, keys, path, csv_path)
            
            
    def handle_three_or_four(self, routine, args):
        
        img_dir, gt_dir, img_files, gt_files, keys, path = args    
        path = os.path.join(self.results_dir, path)
        
        if routine == 3:
            self.eval_pipeline.stage_three(img_dir, gt_dir, img_files, gt_files, keys, path)
        else:
            self.eval_pipeline.stage_four(img_dir, gt_dir, img_files, gt_files, keys, path)     
    
    def evaluate(self, eval_routines):
        
        """This function will run the specified evaluation routines that are present in the evaluation pipeline, it is a helper class.
        eval_routines : dictionary, the keys are numbers from 1-5 specifying which evaluation routines should be run, the values to the keys are the arguments required for that function in the evaluation pipeline"""
        
        for routine in eval_routines.keys():
            
            args = eval_routines[routine]
            if routine == 1:
                
                self.handle_one(args)

            elif routine == 2 or routine == 5:
                
                self.handle_two_or_five(routine, args)            
                    
            elif routine == 3 or routine == 4:
                
                self.handle_three_or_four(routine, args)
            
            else:
                
                print("Invalid routine")
                return -1

                
                    
                                     
                                     
                
                
        
        
    
    