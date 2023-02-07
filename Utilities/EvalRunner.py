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
                self.eval_pipeline.stage_two(model_keys=keys, path=path, csv_path=csv_path)
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

    def quantative(self, stub_name, model_keys):

        s1_ext = stub_name + "_s1.csv"     
        s2_ext = stub_name + "_s2.json"
        s2_csv_ext = stub_name + "_s2.csv"
        s5_ext = stub_name + "_s5.png"
        s5_csv_ext = stub_name + "_s5.csv"

        quant_routine = {

            1 : (model_keys, s1_ext),
            2 : (model_keys, s2_ext, s2_csv_ext),
            5 : (model_keys, s5_ext, s5_csv_ext)
        }

        self.evaluate(quant_routine)
    
    def qualitative(self, stub_name, model_keys, args):

        img_dir, gt_dir, img_files, gt_files = args

        s3_ext = stub_name + "_s3.png"
        s4_ext = stub_name + "_s4.png"

        qual_routine = {
            3: (img_dir, gt_dir, img_files, gt_files, model_keys, s3_ext),
            4 : (img_dir, gt_dir, img_files, gt_files, model_keys, s4_ext)
        }

        self.evaluate(qual_routine)
    
    def all_routines(self, stub_name, model_keys, args):

        self.quantative(stub_name, model_keys)
        self.qualitative(stub_name, model_keys, args)
    
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

                
                    
                                     
                                     
                
                
        
        
    
    