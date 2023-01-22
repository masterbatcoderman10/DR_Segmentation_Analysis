class BaseModel():

    def predict(self, dataloader):
        """This function """  
        self.eval()

        with torch.no_grad():
            
            predictions = []
            for images, _ in dataloader:
                images = images.to(device)
                outputs = self(images)
                predictions.append(outputs.permute(0, 2, 3, 1))
            predictions = torch.cat(predictions, dim=0)
        
        return predictions.numpy()