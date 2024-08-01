from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from typing import Dict
from collections import defaultdict
import torch, ast
from torch import Tensor
import os.path as osp
from jaxtyping import Float, UInt8

class ResampleDataset(InputDataset):
    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        roi_dir = osp.join(osp.dirname(osp.dirname(self._dataparser_outputs.image_filenames[0])), "system_results.txt")
        if osp.exists(roi_dir):
            with open(roi_dir, 'r', encoding='utf-8') as f:
                anno = f.readlines()
            f.close()
            self.img2roi = defaultdict(list)
            for a in anno:
                img, roi = a.rstrip().split('\t')
                roi = ast.literal_eval(roi)
                for r in roi:
                    self.img2roi[img].append(r['points'])
        else:
            self.img2roi = None

    def get_rois(self, image_idx: int) -> Tensor:
        """Returns the rois of shape (num_rois, 4, 2).

        Args:
            image_idx: The image index in the dataset.
        """
        if self.img2roi is None:
            return torch.full((100, 4, 2), -1, dtype=torch.float32)
        
        image_filename = self._dataparser_outputs.image_filenames[image_idx]
        real_rois = self.img2roi[osp.basename(image_filename)]
        if len(real_rois) == 0:
            return torch.full((100, 4, 2), -1, dtype=torch.float32)
        
        rois = torch.full((100, 4, 2), -1, dtype=torch.float32)
        
        rois[:len(real_rois)] = torch.tensor(real_rois, dtype=torch.float32)
        return rois
    
    def get_image_float32(self, image_idx: int) -> Float[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in float32 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        elif image.shape[-1] == 3:
            image = torch.where(image == torch.tensor([0., 0., 0.]), torch.tensor([1.0, 1.0, 1.0]), image)
        return image

    def get_image_uint8(self, image_idx: int) -> UInt8[Tensor, "image_height image_width num_channels"]:
        """Returns a 3 channel image in uint8 torch.Tensor.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx))
        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert (self._dataparser_outputs.alpha_color >= 0).all() and (
                self._dataparser_outputs.alpha_color <= 1
            ).all(), "alpha color given is out of range between [0, 1]."
            image = image[:, :, :3] * (image[:, :, -1:] / 255.0) + 255.0 * self._dataparser_outputs.alpha_color * (
                1.0 - image[:, :, -1:] / 255.0
            )
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        elif image.shape[-1] == 3:
            image = torch.where(image == torch.tensor([0., 0., 0.]), torch.tensor([1.0, 1.0, 1.0]), image)
            image = torch.clamp(image, min=0, max=255).to(torch.uint8)
        return image

    
    def get_metadata(self, data: Dict) -> Dict:
        """Method that can be used to process any additional metadata that may be part of the model inputs.

        Args:
            image_idx: The image index in the dataset.
        """
        rois = self.get_rois(data['image_idx'])
        # if len(rois[rois >= 0]) == 0:
        #     return {}
        # else:
        return {"rois": rois}
        