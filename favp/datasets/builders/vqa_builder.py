"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import warnings

from favp.common.registry import registry
from favp.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from favp.datasets.datasets.slake_vqa import SLAKEVQADataset, SLAKEVQAEvalDataset
from favp.datasets.datasets.dmevqa_dataset import DMEVQADataset,DMEVQAEvalDataset
from favp.datasets.datasets.vqa_rad_dataset import VQARADEvalDataset,VQARADDataset


@registry.register_builder("slake")
class SLAKEVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = SLAKEVQADataset
    eval_dataset_cls = SLAKEVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/slake/slake_vqa.yaml",
    }

@registry.register_builder("dme_vqa")
class DMEVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = DMEVQADataset
    eval_dataset_cls = DMEVQAEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/dme_vqa/dme_vqa.yaml",
    }

@registry.register_builder("vqa_rad")
class VQARADBuilder(BaseDatasetBuilder):
    train_dataset_cls = VQARADDataset
    eval_dataset_cls = VQARADEvalDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/vqa_rad/vqa_rad.yaml",
    }
