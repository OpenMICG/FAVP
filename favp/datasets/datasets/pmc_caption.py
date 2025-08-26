import os
from collections import OrderedDict

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

class PMCCaptionDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root, ann_paths=ann_paths)

        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

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

        img_file = ann["image"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        caption = self.text_processor(ann["caption"])
        instruction = random.choice(self.instruction_pool)
        instruction = "<Img><ImageHere></Img> [caption] {} ".format(instruction)

        return {
            "image": image,
            "answer": caption,
            # "image_id": self.img_ids[ann["image"]],
            "instruction_input": instruction,
        }
