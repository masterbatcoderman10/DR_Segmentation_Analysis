import torch
import numpy as np
from tqdm import tqdm


class BaseModel():
    """
    Base class for segmentation models.

    Args:
        device (torch.device): The device on which the model should run (e.g., 'cuda' or 'cpu').
    """

    def __init__(self, device):

        self.device = device

    def predict_all(self, dataloader) -> np.ndarray:
        """
        This function returns the predictions for all images present in the dataloader.

        Args:
            dataloader (DataLoader): A pytorch DataLoader encapsulating a PyTorch Dataset that returns a batch of images and a target.

        Returns:
            np.ndarray: A numpy array containing segmentation maps for all the images in the dataloader: (n_images, height, width, n_classes).
        """
        self.eval()

        # Assert that the dataloader returns images and targets
        assert len(
            dataloader.dataset[0]) == 2, "The dataloader should return images and targets."

        with torch.no_grad():

            predictions = []
            for images, _ in tqdm(dataloader):
                images = images.to(self.device)
                outputs = self(images)
                outputs = torch.nn.functional.softmax(outputs, dim=1)
                predictions.append(outputs.permute(0, 2, 3, 1))
            predictions = torch.cat(predictions, dim=0)

        return predictions.detach().cpu().numpy()

    def predict_image(self, image) -> np.ndarray:
        """
        This function returns the prediction of a single image.

        Args:
            image (torch.Tensor): The input original image tensor of shape (channels, height, width).

        Returns:
            np.ndarray: Output segmentation map containing class probabilities of shape (height, width, n_classes).
        """
        self.eval()
        with torch.no_grad():

            image = image.unsqueeze(0)
            image = image.to(self.device)
            output = self(image)
            output = torch.nn.functional.softmax(output, dim=1)
            output = output.detach().cpu().numpy()
            output = np.transpose(output, (0, 2, 3, 1))
            output = np.squeeze(output, axis=0)
            return output
