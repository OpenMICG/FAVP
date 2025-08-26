import os
import json
import random
import torch
from PIL import Image, ImageDraw
import numpy as np
import cv2
import torchvision.utils as vutils
import torchvision.transforms as transforms
from favp.datasets.datasets.vqa_datasets import VQADataset, VQAEvalDataset
from favp.datasets.datasets.base_dataset import BaseDataset

from collections import OrderedDict

def get_mask_boundary(mask_image):
    gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_coordinates = []
    for contour in contours:
        for point in contour:
            x, y = point[0]
            boundary_coordinates.append((x, y))

    return boundary_coordinates

def draw_edges_on_image(image, edge_points, color, thickness):
    draw = ImageDraw.Draw(image)
    for i in range(len(edge_points) - 1):
        point1 = tuple(edge_points[i])
        point2 = tuple(edge_points[i + 1])
        draw.line([point1, point2], fill=color, width=thickness)

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


def mask_transform(size):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor()
    ])
    return transform

class DMEVQADataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, mask_root):
        super().__init__(vis_processor, text_processor, vis_root,ann_paths)
        self.mask_root = mask_root
        self.instruction_pool = [
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def get_data(self, index):
        ann = self.annotation[index]
        mask_path = os.path.join(self.mask_root, ann["mask_name"])
        mask = Image.open(mask_path)
        transform = mask_transform(224)
        mask = transform(mask)
        # mask = self.vis_processor(mask)
        image_path = os.path.join(self.vis_root, ann["image_name"])
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)
        new_image = torch.mul(image, mask)

        question = ann["question"]
        # if question == "are there hard exudates in this region?":
        #     question = "are there hard exudates in this circled region?"
        # if question == "are there optic discs in this region?":
        #     question = "are there optic discs in this circled region?"
        question = self.text_processor(question)

        question_id = ann["question_id"]
        answer = ann['answer']
        if type(answer) != str:
            answer = str(answer)

        return {
            # "image": image,
            # "mask": mask,
            # "image_path": image_path,
            # "mask_path": mask_path,
            "image": new_image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)

        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            # "image": data['image'],
            # "image_path": data['image_path'],
            # "mask": data['mask'],
            # "mask_path": data['mask_path'],
            "image": data['image'],
            "question": data['question'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
        }

class DMEVQAEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, mask_root):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """

        self.instruction_pool = [
            #'Question: {} Short answer:',
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
        self.vis_root = vis_root
        self.mask_root = mask_root
        self.ques_file = self.annotation
        self.anno_file = self.annotation

    def get_data(self, index):
        ann = self.annotation[index]

        mask_path = os.path.join(self.mask_root, ann["mask_name"])
        mask = Image.open(mask_path)
        image_path = os.path.join(self.vis_root, ann["image_name"])
        image = Image.open(image_path).convert("RGB")
        # mask = self.vis_processor(mask)
        transform = mask_transform(224)
        mask = transform(mask)
        image = self.vis_processor(image)

        new_image = torch.mul(image, mask)

        question = ann["question"]
        # if question == "are there hard exudates in this region?":
        #     question = "are there hard exudates in this circled region?"
        # if question == "are there optic discs in this region?":
        #     question = "are there optic discs in this circled region?"
        question = self.text_processor(question)
        question_id = ann["question_id"]

        return {
            # "image": image,
            # "mask": mask,
            # "image_path": image_path,
            # "mask_path": mask_path,
            "image": new_image,
            "question": question,
            "question_id": question_id,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            # "image": data['image'],
            # "image_path": data['image_path'],
            # "mask": data['mask'],
            # "mask_path": data['mask_path'],
            "image": data['image'],
            "question": data['question'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
        }