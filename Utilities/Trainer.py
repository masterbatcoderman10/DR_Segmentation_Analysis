class Trainer:

    """This class can be used to train a multi-class segmentation model, it contains many helpful visualization methods that can show the detailed performance of network training."""

    def __init__(self, network, train_dl, epochs, loss_function, optimizer, scheduler=None):

        self.network = network
        self.train_dl = train_dl
        self.epochs = epochs
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler 
    
    def get_mins(self, seconds):
        
        """This function converts seconds to minutes and seconds"""

        return f"{seconds // 60} mins : {seconds % 60} seconds"
    
    def main_step(self, img_batch, target_batch, multi=False):
        
        #Zeroing out previous gradients
        self.optimizer.zero_grad()

        #Make model prediction
        pred = self.network(img_batch)
        
        #Cross-Entropy expects a 1D tensor of the long type
        if multi:
            #Argmax along the channel dimension
            target_batch = torch.argmax(target_batch, dim=1)
            target_batch = target_batch.type(torch.LongTensor).to(device)
        else:
            target_batch = target_batch.float().to(device)
            
        #Compute Loss
        loss = self.loss_function(pred, target_batch)
        #Calculate gradients through backpropagation
        loss.backward()            
        #Update the model parameters
        self.optimizer.step()
        
        return loss, pred
    
    def eval_step(self,img_batch, target_batch, multi=False):
    
        #Assumes the network is already put into evaluation mode
        val_pred = self.network(img_batch)
        #Cross-Entropy expects a 1D tensor of the long type
        if multi:
            #Argmax along the channel dimension
            target_batch = torch.argmax(target_batch, dim=1)
            target_batch = target_batch.type(torch.LongTensor).to(device)
        else:
            target_batch = target_batch.float().to(device)

        #Compute Loss
        loss = self.loss_function(val_pred, target_batch)

        return loss, val_pred
    
    def plot_sample_prediction(self, img_batch, target_batch, pred_batch, ix, n_classes=1, background=False):
    
        """This function plots along with a sample image and annotation, the prediction for the sample image
        """
        assert n_classes > 0
        multi = False if n_classes == 1 else True
        #Initializing the softmax and the sigmoid
        softmax = nn.Softmax(dim=0)
        sigmoid = nn.Sigmoid()
        
        fig, ax = plt.subplots(1,4, figsize=(8,4))

        #Getting an image and reshaping it
        test_img = img_batch[ix]
        test_img = torch.reshape(test_img, (512,512,3))
        #Removing the image from the gpu and the computation graph
        n_img = test_img.to('cpu').detach().numpy()
        
        H = n_img.shape[0]
        W = n_img.shape[1]
        
        #Getting the corresponding annotation
        #For multi-class it will be (C,H,W)
        #Removing the annotation from the gpu and computation graph
        test_ann = target_batch[ix]
        n_ann = test_ann.to('cpu').detach().numpy()
        #Making channels last for a single annotation
        n_ann = np.rollaxis(n_ann, 0, 3)
        
        #Getting the corresponding prediction
        #For multi-class it will be (C,H,W)
        test_pred = pred_batch[ix]
        if multi:
            #applying the softmax function to the prediction since they are just scores
            test_pred = softmax(test_pred)
        else:
            #applying the sigmoid function to the prediction
            test_pred = sigmoid(test_pred)
        
        #Thresholded prediction
        threshold = torch.nn.Threshold(0.75, 0)
        test_pred_clamped = threshold(test_pred)
        #For multiclass it will be (C,W,H)
        n_pred = test_pred.to('cpu').detach().numpy()
        #n_pred = np.rollaxis(n_pred, 0, 3)
        
        #For multiclass it will be (C,W,H)
        n_pred_clamped = test_pred_clamped.to("cpu").detach().numpy()
        #n_pred_clamped = np.rollaxis(n_pred_clamped, 0, 3)
        
        
        #Creating masks that can be plotted
        if not background and multi:
            #This step creates a mask for the background and concatenates it to the front of the annotation and prediction
            #Then the argmax operation is performed to obtain a matrix which can be plotted
            
            bg = np.full((H,W, 1), 0.1)
            n_ann = np.concatenate([bg, n_ann], axis=-1)
            n_ann = np.argmax(n_ann, axis=-1)

            n_pred = np.concatenate([bg, n_pred], axis=0)
            n_pred = np.argmax(n_pred, axis=0)
            
            n_pred_clamped = np.concatenate([bg, n_pred_clamped], axis=0)
            n_pred_clamped = np.argmax(n_pred_clamped, axis=0)
            
            
        elif background and multi:
            
            n_ann = np.argmax(n_ann, axis=-1)
            n_pred = np.argmax(n_pred, axis=0)
            n_pred_clamped = np.argmax(n_pred_clamped, axis=0)
        else:
            
            n_ann = n_ann
            n_pred = n_pred
            n_pred_clamped = n_pred_clamped
        

        #Plotting the image
        ax[0].imshow(n_img)
        ax[0].axis("off")
        #Plotting the annotation
        ax[1].imshow(n_ann)
        ax[1].axis("off")
        #Plotting the prediction
        ax[2].imshow(n_pred)
        ax[2].axis("off")
        #Plotting a thresholded prediction
        ax[3].imshow(n_pred_clamped)
        ax[3].axis("off")
        plt.show()

    def plot_class_activations(self, target_batch, pred_batch, n_classes):
    
        """This function plots the individual class activations given a prediction image. 
        This function assumes that the channels are first.
        This function also assumes that no softmax has been applied
        """
        
        softmax = nn.Softmax(dim=0)
        
        test_pred = pred_batch[0]
        test_ann = target_batch[0].detach().to("cpu")
        
        soft_pred = softmax(test_pred)
        soft_pred = soft_pred.detach().to("cpu").numpy()
        
        fig, ax = plt.subplots(1, n_classes, figsize=(n_classes*2, n_classes))
        
        for i in range(n_classes):
            
            ax[i].imshow(test_ann[i])
            ax[i].axis("off")
        
        fig, ax2 = plt.subplots(1, n_classes, figsize=(n_classes*2,n_classes))
        
        for i in range(n_classes):

            ax2[i].imshow(soft_pred[i])
            ax2[i].axis("off")
        plt.show()

    def fit(self, log=True, validation=False, valid_dl=None):

        """Calling this function initiates network training.
        Arguments: log: True or False, defaults to True. This argument determines whether to log the loss at the end of the epoch along with the time taken.
                   validation: True or False, defaults to False. This argument determines whether a validation step should be taken. If this is set to true, a validation dataloader must be passed in.
                   valid_dl: A torch dataloader containing the same data type as the training dataloader must be passed in, if validation is set to True. 
        """

        for e in range(self.epochs):
            print(f"Starting epoch : {e+1} -------------------------------------------------------------------")
            elapsed_time = 0
            st = time.time()
            loss_value = 0
            
            #Indicates start of batch
            start = True
            start_2 = True
            total_batches = 0
            
            #Training Loop
            self.network.train()
            for img_batch, annotation_batch in self.train_dl:
                
                total_batches += 1
                #Putting the images and annotations on the device
                img_batch = img_batch.to(device)
                #Obtaining the loss and the predictions for current batch - This is multiclass classification
                loss, pred = self.main_step(img_batch, annotation_batch, multi=True)
            
                #Check for the start of the batch to visualize a prediction
                if start:
                    self.plot_sample_prediction(img_batch, annotation_batch, pred, 0, 4, True)
                    #Indicate that next batch is not start of epoch
                    print(f"Plotting Activations")
                    self.plot_class_activations(annotation_batch.to(device), pred, 5)
                    start = False
                
                #Updating loss by adding loss for current batch  
                loss_value += loss.item()
                if start_2:
                    print(f"The loss on the first batch is : {loss_value}")
                    start_2 = False
            
            #If logging is enabled print total loss value for the epoch divided by batch size
            if log:
                loss_for_epoch = round(loss_value / total_batches, 3)
                print(f"Loss at epoch : {e+1} : {loss_for_epoch}")

            
            #Modifying learning rate
            if self.scheduler is not None:
                self.scheduler.step()
            
            #Validation Loop
            ######################################################################################################################################
            if validation and valid_dl is not None:

                print("Running Validation Step")
                ######### Validation step ############
                val_loss = 0
                val_start = True
                val_start_2 = True
                val_batches = 0
                with torch.no_grad():
                    self.network.eval()

                    for img_batch, annotation_batch in valid_dl:
                        
                        val_batches += 1
                        val_img_batch = img_batch.to(device)
                        valid_loss, val_pred = self.eval_step(val_img_batch, annotation_batch, multi=True)
                        
                        if val_start:
                            self.plot_sample_prediction(val_img_batch, annotation_batch, val_pred, 0, 4, True)
                            val_start = False

                        val_loss += valid_loss.item()

                        if val_start_2:
                            print(f"The loss on the first batch for validation is : {val_loss}")
                            val_start_2 = False


                #If logging is enabled print total loss value for the epoch divided by batch size
                if log:
                    val_loss_for_epoch = round(val_loss / val_batches, 3)
                    print(f"Validation Loss at epoch : {e+1} : {val_loss_for_epoch}")

                
            #End of Epoch -----------------------------------------------------------------------------------------------------------------------
            #Calculate the end time and log
            et = time.time()
            elapsed_time = et - st
            print(f"Epoch : {e+1} took {self.get_mins(elapsed_time)}")
            print("\n")
                
            ######### End of validation step #######
            print("------------------------------------------------------------------------------------------")
            print("\n")
            print("\n")
            print("\n")
                
                
                    

                    
