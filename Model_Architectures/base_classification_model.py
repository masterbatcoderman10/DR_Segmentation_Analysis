class BaseClassificationModel:

    def predict(self, dataloader):

        """This function returns the predicted class probabilities for each image present in the dataloader"""

        self.eval()

        with torch.no_grad():
            
            predictions = []
            for images, _ in tqdm(dataloader):
                images = images.to(device)
                outputs = self(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(outputs)
            predictions = torch.cat(predictions, dim=0)
        
        return predictions.detach().cpu().numpy()