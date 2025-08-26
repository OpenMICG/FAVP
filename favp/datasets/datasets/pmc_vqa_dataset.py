import os
import json
import random

from PIL import Image

from favp.datasets.datasets.vqa_datasets import VQADataset

from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["Figure_path"],
                "question": ann["Question"],
               # "question_id": ann["question_id"],
                "answers": ann["Answer"],
                "image": sample["image"],
            }
        )


class PMCVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root,ann_paths=ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["Figure_path"])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["Figure_path"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)

        question = self.text_processor(ann["Question"])
        # question_id = ann["question_id"]
        answer = self.text_processor(ann["Answer"])

        return {
            "image": image,
            "question": question,
            # "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            # "question_id": data["question_id"],
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['answer'],
        }
