from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor

from einops import rearrange
import uuid
import os

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    ImageProjModel,
)
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False

    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)

        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

    def get_image_proj(
        self,
        image_prompt: Tensor,
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values
        image_prompt = image_prompt.to(self.image_encoder.device)
        image_prompt_embeds = self.image_encoder(
            image_prompt
        ).image_embeds.to(
            device=self.device, dtype=torch.bfloat16,
        )
        # encode image
        image_proj = self.improj(image_prompt_embeds)
        return image_proj

    def __call__(self,
                 prompt: str,
                 image_prompt: Image = None,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        if not (image_prompt is None and neg_image_prompt is None) :
            assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

            if image_prompt is None:
                image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
            if neg_image_prompt is None:
                neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)

            image_proj = self.get_image_proj(image_prompt)
            neg_image_proj = self.get_image_proj(neg_image_prompt)

        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height)
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(
                2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_proj,
            neg_image_proj=neg_image_proj,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            if self.controlnet_loaded:
                x = denoise_controlnet(
                    self.model,
                    **inp_cond,
                    controlnet=self.controlnet,
                    timesteps=timesteps,
                    guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    controlnet_gs=control_weight,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                )
            else:
                x = denoise(
                    self.model,
                    **inp_cond,
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()


class XFluxSampler(XFluxPipeline):
    def __init__(self, clip, t5, ae, model, device):
        self.clip = clip
        self.t5 = t5
        self.ae = ae
        self.model = model
        self.model.eval()
        self.device = device
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.offload = False
