import os
from collections import OrderedDict
import cv2
import torch

from favp.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
import random

class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )

class RocoCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root, ann_paths=ann_paths)
        # self.ann_pretrain = []

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann['PMC_ID'])
            if os.path.exists(image_path) and image_path.lower().endswith(".jpg"):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

        self.instruction_pool = [
            'describe this image.',
            'Provide a depiction of this image.',
            'Summarize this image .',
            'A image caption:',
            'A image description:',
            'An image that shows ',
            'Write a description for the image.',
        ]

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["PMC_ID"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
            # "image_path": image_path,
            "image": image,
            "answer": caption,
            "instruction_input": instruction,
        }
