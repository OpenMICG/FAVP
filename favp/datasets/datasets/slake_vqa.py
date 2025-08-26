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
            # "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def get_data(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vis_root, ann["img_name"])

        # ori_image = Image.open(image_path).convert('RGB')
        # path = os.path.join("/data3/tongzixuan/dataset/Slake/Slakenew/seg/train",ann["img_name"].split('/')[0])
        # image = []
        # if not os.path.exists(path):
        #     image.append(self.vis_processor(ori_image))
        # else:
        #     seg_path = []
        #     seg_name = os.listdir(path)
        #     for sn in seg_name:
        #         seg_path.append(os.path.join(path, sn))
        #     seg_images = []
        #     seg_images_t = []
        #     for seg_p in seg_path:
        #         seg_images.append(Image.open(seg_p).convert('RGB'))
        #     # ori_image = self.vis_processor(ori_image)
        #     image.append(self.vis_processor(ori_image))
        #     for seg_image in seg_images:
        #         seg_images_t.append(self.vis_processor(seg_image))
        #     # seg_images_t = torch.cat(seg_images_t, dim=0)
        #     # image = torch.cat([ori_image, seg_images_t], dim=0)
        #     image.extend(seg_images_t)

        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["qid"]
        answer = self.text_processor(ann['answer'])
        # print("image", image.shape)
        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            # "image_path": image_path,
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
            # "image_path": data['image_path']
        }


class SLAKEVQAEvalDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor=vis_processor, text_processor=text_processor, vis_root=vis_root, ann_paths=ann_paths)

        self.instruction_pool = [
            #'Question: {} Short answer:',
            # "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]
        self.vis_root = vis_root
        # for ann_path in ann_paths:
        #     self.annotation.append(json.load(open(ann_path)))
        self.ques_file = self.annotation
        self.anno_file = self.annotation

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["img_name"])
        # ori_image = Image.open(image_path).convert('RGB')
        # image = []
        # if self.vis_root.split('/')[-1] == 'val':
        #     path = os.path.join("/data3/tongzixuan/dataset/Slake/Slakenew/seg/val", ann["img_name"].split('/')[0])
        # else:
        #     path = os.path.join("/data3/tongzixuan/dataset/Slake/Slakenew/seg/test", ann["img_name"].split('/')[0])
        # if not os.path.exists(path):
        #     image.append(self.vis_processor(ori_image))
        # else:
        #     seg_path = []
        #     seg_name = os.listdir(path)
        #     for sn in seg_name:
        #         seg_path.append(os.path.join(path, sn))
        #     seg_images = []
        #     seg_images_t = []
        #     for seg_p in seg_path:
        #         seg_images.append(Image.open(seg_p).convert('RGB'))
        #     ori_image = self.vis_processor(ori_image)
        #     for seg_image in seg_images:
        #         seg_images_t.append(self.vis_processor(seg_image))
        #     # seg_images_t = torch.cat(seg_images_t, dim=0)
        #     # image = torch.cat([ori_image, seg_images_t], dim=0)
        #     image.extend(seg_images_t)
        # image = cv2.imread(image_path)
        # image = Image.open(image_path).convert("RGB")
        # image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        return {
            "image": image,
            # 'image_path': image_path,
            "question": question,
            "question_id": ann["qid"],
            "instruction_input": instruction,
        }
