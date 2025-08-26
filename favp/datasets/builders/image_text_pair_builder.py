import os
import logging
import warnings

from favp.common.registry import registry

from favp.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from favp.datasets.datasets.pmc_caption import PMCCaptionDataset
from favp.datasets.datasets.pmc_vqa_dataset import PMCVQADataset
from favp.datasets.datasets.roco_caption_dataset import RocoCaptionDataset


@registry.register_builder("pmc_caption")
class PMCCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pmc/pmc_caption.yaml",
    }

@registry.register_builder("pmc_vqa")
class PMCVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = PMCVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/pmc/pmc_vqa.yaml",
    }


@registry.register_builder("roco_caption")
class RocoCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = RocoCaptionDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/roco/roco_caption.yaml",
    }