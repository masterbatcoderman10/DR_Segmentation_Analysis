class BaseModel():

    def predict(self, dataloader):
        """This function returns the predictions for all images present in the dataloader"""
        self.eval()

        with torch.no_grad():
            
            predictions = []
            for images, _ in tqdm(dataloader):
                images = images.to(device)
                outputs = self(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(outputs.permute(0, 2, 3, 1))
            predictions = torch.cat(predictions, dim=0)
        
        return predictions.detach().cpu().numpy()
    
    def predict_image(self, image):
        """This function returns the prediction of one image"""
        self.eval()
        with torch.no_grad():
            
            image = image.unsqueeze(0)
            image = image.to(device)
            output = self(image)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.detach().cpu().numpy()
            output = np.transpose(output, (0, 2, 3, 1))
            return output