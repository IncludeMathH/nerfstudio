from nerfstudio.data.datasets.base_dataset import InputDataset
from typing import Dict
import cv2, torch
import numpy as np

class EdgeDataset(InputDataset):
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        image = (data['image'].numpy() * 255).astype('uint8')
        # Convert the image to grayscale
        # gray_image = cv2.cvtColor(image * 255, cv2.COLOR_RGB2GRAY)

        # 使用Canny边缘检测
        edges = cv2.Canny(image, 50, 100)

        data["edge"] = torch.from_numpy(edges.astype(np.float32) / 255)
        
        return data
        