import os
import json
import random

from PIL import Image
import torch
from favp.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from favp.datasets.datasets.base_dataset import BaseDataset

from collections import OrderedDict
import cv2


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["img_name"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "answers": ann["answer"],
                "image": sample["image"],
            }
        )


class SLAKEVQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root,ann_paths=ann_paths )

        self.instruction_pool = [
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["img_name"])

        question = self.text_processor(ann["question"])
        question_id = ann["qid"]
        answer = self.text_processor(ann['answer'])

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            "image_path": image_path,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": data['answer'],
            "image_path": data['image_path']
        }


class SLAKEVQAEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root, ann_paths=ann_paths)

        self.instruction_pool = [
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
        self.vis_root = vis_root
        self.ques_file = self.annotation
        self.anno_file = self.annotation

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_name"])

        question = self.text_processor(ann["question"])
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            'image_path': image_path,
            "question": question,
            "question_id": ann["qid"],
            "instruction_input": instruction,
        }
