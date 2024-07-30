from nerfstudio.data.datasets.edge_dataset import EdgeDataset
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager, ParallelDataManagerConfig
from dataclasses import dataclass, field
from nerfstudio.data.datamanagers.base_datamanager import TDataset

from typing import Type, cast, get_origin, get_args, ForwardRef
from functools import cached_property
from nerfstudio.utils.misc import get_orig_class

@dataclass
class ParallelEdgeDataManagerConfig(ParallelDataManagerConfig):
    """Config for a `ParallelDataManager` which reads data in multiple processes"""

    _target: Type = field(default_factory=lambda: ParallelEdgeDataManager)
    """Target class to instantiate."""


class ParallelEdgeDataManager(ParallelDataManager):
    @cached_property
    def dataset_type(self) -> Type[InputDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, EdgeDataset)  # type: ignore
        orig_class: Type[ParallelDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is ParallelDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is ParallelDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is ParallelDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default