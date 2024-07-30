from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from typing import Dict
from collections import defaultdict
import torch, ast
from torch import Tensor
import os.path as osp

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
        