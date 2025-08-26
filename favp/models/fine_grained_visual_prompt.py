from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from .transform import bias_crop, boxes_to_circles, scale_contour, scale_mask, str2rgb

class FGVP_ENSEMBLE:
    def __init__(
        self,
        color_line,
        thickness,
        color_mask,
        alpha,
        vit_processing,
        vit_image_size,
        resize_transform_vit,
        pixel_mean,
        pixel_std,
        blur_std_dev,
        mask_threshold=0.0,
        contour_scale=1.0,
        device='cpu',
    ):
        self.color_line = color_line
        self.thickness = thickness
        self.color_mask = color_mask
        self.alpha = alpha
        self.vit_processing = vit_processing
        self.vit_image_size = vit_image_size
        self.resize_transform_vit = resize_transform_vit
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.blur_std_dev = blur_std_dev
        self.mask_threshold = mask_threshold
        self.contour_scale = contour_scale
        self.device = device

    def __call__(self, visual_prompt: str, image, centers, boxes, masks):
        # image    np.array(H, W, 3) uint8, rgb[0~255]
        # centers  torch.Tensor(N, 2) float
        # boxes    torch.Tensor(N, 4) float
        # masks    torch.Tensor(N, 1, H, W) bool,
        assert len(centers) == len(boxes) == len(masks)
        ori_size = image.shape[:2]
        image = self.resize_transform_vit.apply_image(image)
        new_size = image.shape[:2]
        vit_inputs = []
        vit_inputs.append(image)
        if len(masks) == 0:
            vit_inputs = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            vit_inputs = (vit_inputs - self.pixel_mean) / self.pixel_std
            vit_inputs = F.interpolate(vit_inputs, (self.vit_image_size, self.vit_image_size),
                                        mode='bilinear', align_corners=False)
            return vit_inputs
        else:
            masks = F.interpolate(masks.float(), (new_size[0], new_size[1]),
                                  mode="bilinear", align_corners=False)  # N, 1, H, W
            bit_masks = masks > self.mask_threshold
            boxes = self.resize_transform_vit.apply_boxes_torch(boxes, ori_size)
            boxes_centers, axes_lengths = boxes_to_circles(boxes, *new_size)
            # for mask in bit_masks:
            for center, mask, box, boxes_center, axes_length in zip(centers, bit_masks, boxes, boxes_centers,axes_lengths):
                res = image.copy()
                center = center.cpu().numpy()
                mask = mask.squeeze(0).cpu().numpy()
                box = box.int().cpu().numpy()
                boxes_center = boxes_center.cpu().numpy()
                axes_length = axes_length.cpu().numpy()
                if 'mask' == visual_prompt:
                    overlay = res.copy()
                    overlay[mask == 1] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'grayscale_mask' == visual_prompt:
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[mask == 0] = gray[mask == 0]
                elif 'reverse_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay[mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'blur_mask' == visual_prompt:
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(scale_mask(mask, self.contour_scale)))
                    res = np.array(overlay)
                elif 'contour' == visual_prompt:
                    contours, hierarchy = cv2.findContours(mask.astype(
                        np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if self.contour_scale == 1:
                        scaled_contours = contours
                    else:
                        scaled_contours = [scale_contour(cnt, self.contour_scale) for cnt in contours]
                    res = cv2.drawContours(res, scaled_contours, contourIdx=-1,
                                           color=str2rgb(self.color_line), thickness=self.thickness)
                elif 'keypoint' == visual_prompt:
                    res = cv2.circle(res, center.astype(int), int(0.06 * self.vit_image_size),
                                     color=str2rgb(self.color_line), thickness=self.thickness)
                elif 'circle_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay = cv2.ellipse(overlay, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                          color=str2rgb(self.color_mask), thickness=-1)
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'grayscale_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[circle_mask == 0] = gray[circle_mask == 0]
                elif 'reverse_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    overlay = res.copy()
                    overlay[circle_mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'blur_circle_mask' == visual_prompt:
                    circle_mask = np.zeros(res.shape, dtype=np.uint8)
                    circle_mask = cv2.ellipse(circle_mask, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                              color=(255, 255, 255), thickness=-1)[:, :, 0]
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(scale_mask(circle_mask, self.contour_scale)))
                    res = np.array(overlay)
                elif 'circle' == visual_prompt:
                    res = cv2.ellipse(res, boxes_center.astype(int), axes_length.astype(int), 0, 0, 360,
                                      color=str2rgb(self.color_line), thickness=self.thickness)
                elif 'box_mask' == visual_prompt:
                    overlay = res.copy()
                    overlay = cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]),
                                            color=str2rgb(self.color_mask), thickness=-1)
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'grayscale_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    gray = cv2.cvtColor(res.copy(), cv2.COLOR_BGR2GRAY)[:, :, None].repeat(3, -1)
                    res[box_mask == 0] = gray[box_mask == 0]
                elif 'reverse_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    overlay = res.copy()
                    overlay[box_mask == 0] = np.array(str2rgb(self.color_mask))
                    res = cv2.addWeighted(overlay, self.alpha, res, 1 - self.alpha, 0.0)
                elif 'blur_box_mask' == visual_prompt:
                    box_mask = np.zeros(res.shape, dtype=np.uint8)
                    box_mask = cv2.rectangle(box_mask, (box[0], box[1]), (box[2], box[3]),
                                             color=(255, 255, 255), thickness=-1)[:, :, 0]
                    res = Image.fromarray(res)
                    overlay = res.copy()
                    overlay = overlay.filter(ImageFilter.GaussianBlur(self.blur_std_dev))
                    overlay.paste(res, mask=Image.fromarray(box_mask))
                    res = np.array(overlay)
                elif 'box' == visual_prompt:
                    res = cv2.rectangle(res, (box[0], box[1]), (box[2], box[3]),
                                        color=str2rgb(self.color_line), thickness=self.thickness)
                else:
                    raise ValueError("Without this Prompt")
            # rgb -> normalized
                vit_inputs.append(res)
            vit_inputs = np.stack(vit_inputs)
            vit_inputs = torch.from_numpy(vit_inputs).float().permute(0, 3, 1, 2).to(self.device)
            vit_inputs = (vit_inputs - self.pixel_mean) / self.pixel_std
            if self.vit_processing == 'padding':
                # Pad
                h, w = vit_inputs.shape[-2:]
                padh = self.vit_image_size - h
                padw = self.vit_image_size - w
                vit_inputs = F.pad(vit_inputs, (0, padw, 0, padh))
            elif self.vit_processing == 'bias_crop':
                vit_inputs = torch.cat([bias_crop(c.unsqueeze(0), (self.vit_image_size, self.vit_image_size), b)
                                        for c, b in zip(vit_inputs, boxes)], 0)
            elif self.vit_processing == 'resize':
                vit_inputs = F.interpolate(vit_inputs, (self.vit_image_size, self.vit_image_size),
                                            mode='bilinear', align_corners=False)
            else:
                raise NotImplementedError
        # return nomalized inputs (N, 3, H, W)
        # print("vit_input", vit_inputs.size())

        return vit_inputs

