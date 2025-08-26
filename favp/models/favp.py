import logging
import random
import cv2

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import numpy as np
from favp.common.registry import registry
from favp.models.base_model import disabled_train
from favp.models.favp_base import FAVPBase
from favp.models.Qformer import BertConfig, BertLMHeadModel


from favp.segmed.utils.transforms import ResizeLongestSide
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from favp.models.fine_grained_visual_prompt import FGVP_ENSEMBLE
from favp.segmed.build_sam import sam_model_registry
from favp.segmed.automatic_mask_generator import SamAutomaticMaskGenerator


@registry.register_model("favp")
class FAVP(FAVPBase):
    """
    MiniGPT-4 model
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/vicuna0.yaml",
        "pretrain_llama2": "configs/models/llama2.yaml",
    }

    def __init__(
            self,
            vit_model="eva_clip_g",
            q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
            img_size=512,drop_path_rate=0,use_grad_checkpoint=False,vit_precision="fp16",freeze_vit=True,has_qformer=True,freeze_qformer=True,
            num_query_token=32,llama_model="",prompt_path="",prompt_template="",max_txt_len=32,end_sym='###',low_resource=False,
            device_8bit=0, lora_r=0,lora_alpha=4,setting="",chat_template=False,points_per_side=50,points_per_batch=256,
            pred_iou_thresh=0.68,stability_score_thresh=0.7,stability_score_offset=0.7,box_nms_thresh=0.5,crop_n_layers=0,crop_nms_thresh=0.7,
            crop_overlap_ratio=512/1500,crop_n_points_downscale_factor=2,point_grids=None,min_mask_region_area=400,output_mode='binary_mask',
            prompt="mask",color_line='red',thickness=2,color_mask='green',alpha=0.5,vit_processing='resize',vit_image_size=224,contour_scale=1.0,
            device='cuda',blur_std_dev=100,has_sam=False
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            setting=setting,
            lora_alpha = lora_alpha
        )

        self.has_sam = has_sam
        self.chat_template = chat_template
        self.has_qformer = has_qformer
        self.prompt = prompt
        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                num_query_token, self.visual_encoder.num_features, freeze_qformer
            )
            self.load_from_pretrained(url_or_filename=q_former_model)  # load q-former weights here

            img_f_dim = self.Qformer.config.hidden_size
            print('Loading Q-Former Done')
        else:
            img_f_dim = self.visual_encoder.num_features * 4
            print('Do not use Q-Former here.')

        if self.has_sam:
            self.sam_mask_generator, self.fgvp = self.init_SamMed(
                points_per_side,points_per_batch, pred_iou_thresh,
                stability_score_thresh,stability_score_offset,box_nms_thresh,crop_n_layers,crop_nms_thresh,
                crop_overlap_ratio,crop_n_points_downscale_factor,point_grids,min_mask_region_area,
                output_mode,color_line,thickness,color_mask,alpha,vit_processing,vit_image_size,
                contour_scale,device,blur_std_dev)
        else:
            print('Do not use SAM-Med here.')


        self.llama_proj = nn.Linear(
            img_f_dim, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("/data3/tongzixuan/checkpoints/bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    @classmethod
    def init_SamMed(
            self, points_per_side, points_per_batch, pred_iou_thresh, stability_score_thresh,
            stability_score_offset, box_nms_thresh, crop_n_layers, crop_nms_thresh, crop_overlap_ratio,
            crop_n_points_downscale_factor, point_grids, min_mask_region_area, output_mode, color_line, thickness,
            color_mask, alpha, vit_processing, vit_image_size, contour_scale, device, blur_std_dev
    ):
        logging.info("loading SamMed")
        sam_model = sam_model_registry["vit_b"](image_size=256,
                                                checkpoint="/data3/tongzixuan/checkpoints/sammed/sam-med2d_b.pth",
                                                encoder_adapter=True).to(device)
        for name, param in sam_model.named_parameters():
            param.requires_grad = False
        sam_model = sam_model.eval()
        sam_model.train = disabled_train
        sam_mask_generator = SamAutomaticMaskGenerator(
            sam_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,
            box_nms_thresh=box_nms_thresh,
            crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,
            crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,
            point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,
            output_mode=output_mode,
        )
        logging.info("freeze SamMed")

        resize_transform_vit = ResizeLongestSide(vit_image_size)
        logging.info("loading PROMPT")
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).view(-1, 1, 1).to(device) * 255.0
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).view(-1, 1, 1).to(device) * 255.0
        fgvp = FGVP_ENSEMBLE(
            color_line=color_line,
            thickness=thickness,
            color_mask=color_mask,
            alpha=alpha,
            vit_processing=vit_processing,
            vit_image_size=vit_image_size,
            resize_transform_vit=resize_transform_vit,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            blur_std_dev=blur_std_dev,
            mask_threshold=sam_model.mask_threshold,
            contour_scale=contour_scale,
            device=device,
        )
        logging.info("loading PROMPT Done")

        return sam_mask_generator, fgvp

    def get_masks(self,image):
        outputs = self.sam_mask_generator.generate(image)
        masks = []
        boxes = []
        for x in outputs:
            masks.append(x['segmentation'])
            boxes.append(x['bbox'])
        if len(masks) == 0:
            return masks, boxes
        else:
            masks = torch.from_numpy(np.stack(masks)).unsqueeze(1)

            boxes = torch.from_numpy(np.stack(boxes)).float()
            boxes[:, 2] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 1]

            return masks, boxes

    def encode_sam_out(self, image_path):
        image = cv2.imread(image_path[0])

        masks, boxes = self.get_masks(image)
        if len(boxes) == 0:
            centers = []
            vit_inputs = self.fgvp(self.prompt, image[:, :, ::-1], centers, boxes, masks)
        else:
            centers = torch.stack([boxes[:, 0::2].mean(1), boxes[:, 1::2].mean(1)], 1)
            vit_inputs = self.fgvp(self.prompt, image[:, :, ::-1],  centers, boxes, masks)

        with self.maybe_autocast():
            image_list = []
            if vit_inputs.shape[0] > 1:
                for patch in torch.split(vit_inputs, 1, dim=0):
                    image_list.append(self.ln_vision(self.visual_encoder(patch)).to("cuda"))
                local_feats = torch.cat(image_list[1:], dim=1)
                global_feats = image_list[0]
                image_embeds = torch.cat([local_feats, global_feats], dim=1)
            else:
                image_embeds = self.ln_vision(self.visual_encoder(vit_inputs)).to("cuda")
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to("cuda")

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
                inputs_llama = self.llama_proj(image_embeds)

            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to("cuda")

        return inputs_llama, atts_llama

    def encode_img(self, image):
        device = image.device

        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])

        with self.maybe_autocast():
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if self.has_qformer:
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
                query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )

                inputs_llama = self.llama_proj(query_output.last_hidden_state)
            else:
                image_embeds = image_embeds[:, 1:, :]
                bs, pn, hs = image_embeds.shape
                image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))

                inputs_llama = self.llama_proj(image_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image.device)
        return inputs_llama, atts_llama

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model",
                                 "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '###')
        setting = cfg.get("setting", '')
        lora_r = cfg.get("lora_r", 0)
        lora_alpha = cfg.get("lora_alpha", 8)
        chat_template = cfg.get("chat_template", False)
        points_per_side = cfg.get("points_per_side", 36)
        points_per_batch = cfg.get("points_per_batch", 256)
        pred_iou_thresh = cfg.get("pred_iou_thresh", 0.68)
        stability_score_thresh = cfg.get("stability_score_thresh", 0.86)
        stability_score_offset = cfg.get("stability_score_offset", 0.7)
        box_nms_thresh = cfg.get("box_nms_thresh", 0.5)
        crop_n_layers = cfg.get("crop_n_layers", 0)
        crop_nms_thresh = cfg.get("crop_nms_thresh", 0.7)
        crop_overlap_ratio = cfg.get("crop_overlap_ratio", 512 / 1500)
        crop_n_points_downscale_factor = cfg.get("crop_n_points_downscale_factor", 2)
        point_grids = cfg.get("point_grids", None)
        min_mask_region_area = cfg.get("min_mask_region_area", 400)
        output_mode = cfg.get("output_mode", "binary_mask")
        prompt = cfg.get("prompt", "mask")
        color_line = cfg.get("color_line", 'red')
        thickness = cfg.get("thickness", 2)
        color_mask = cfg.get("color_mask", 'green')
        alpha = cfg.get("alpha", 0.5)
        vit_processing = cfg.get("vit_processing", 'resize')
        vit_image_size = cfg.get("vit_image_size", 224)
        contour_scale = cfg.get("contour_scale", 1.0)
        device = cfg.get("device", "cuda")
        blur_std_dev = cfg.get("blur_std_dev", 100)
        has_sam = cfg.get("has_sam", False)

        model = cls(
            vit_model=vit_model,q_former_model=q_former_model,img_size=img_size,drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,vit_precision=vit_precision,freeze_vit=freeze_vit,
            has_qformer=has_qformer,freeze_qformer=freeze_qformer,num_query_token=num_query_token,llama_model=llama_model,
            prompt_path=prompt_path,prompt_template=prompt_template,max_txt_len=max_txt_len,end_sym=end_sym,
            low_resource=low_resource,device_8bit=device_8bit,lora_r=lora_r,lora_alpha=lora_alpha,setting=setting,
            chat_template=chat_template,points_per_side=points_per_side,points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,stability_score_thresh=stability_score_thresh,
            stability_score_offset=stability_score_offset,box_nms_thresh=box_nms_thresh,crop_n_layers=crop_n_layers,
            crop_nms_thresh=crop_nms_thresh,crop_overlap_ratio=crop_overlap_ratio,
            crop_n_points_downscale_factor=crop_n_points_downscale_factor,point_grids=point_grids,
            min_mask_region_area=min_mask_region_area,output_mode=output_mode,prompt=prompt,color_line=color_line,
            thickness=thickness,color_mask=color_mask,alpha=alpha,vit_processing=vit_processing,
            vit_image_size=vit_image_size,contour_scale=contour_scale,device=device, blur_std_dev=blur_std_dev,has_sam=has_sam
        )

        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print("Load MiniGPT-4 Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
