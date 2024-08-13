from PIL import Image
import numpy as np
import torch

from einops import rearrange

from src.flux.modules.layers import DoubleStreamBlockLoraProcessor
from src.flux.sampling import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (load_ae, load_clip, load_flow_model, load_t5, load_controlnet,
                           load_flow_model_quintized, Annotator, get_lora_rank, load_checkpoint)


class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False, seed: int = None):
        self.device = torch.device(device)
        self.offload = offload
        self.seed = seed
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False

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
            lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight
            lora_attn_procs[name].load_state_dict(lora_state_dict)
            lora_attn_procs[name].to(self.device)

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device).to(torch.bfloat16)

        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)

        if control_type == "depth":
            self.controlnet_gs = 0.9
        else:
            self.controlnet_gs = 0.7
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True

    def __call__(self,
                 prompt: str,
                 controlnet_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 true_gs = 3,
                 neg_prompt: str = '',
                 timestep_to_start_cfg: int = 0,
                 ):
        width = 16 * width // 16
        height = 16 * height // 16
        if self.controlnet_loaded:
            controlnet_image = self.annotator(controlnet_image, width, height)
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)

        return self.forward(prompt, width, height, guidance, num_steps, controlnet_image,
         timestep_to_start_cfg=timestep_to_start_cfg, true_gs=true_gs, neg_prompt=neg_prompt)

    def forward(self, prompt, width, height, guidance, num_steps, controlnet_image=None, timestep_to_start_cfg=0, true_gs=3, neg_prompt=""):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=self.seed
        )
        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(self.seed)
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
                    self.model, **inp_cond, controlnet=self.controlnet,
                    timesteps=timesteps, guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    controlnet_gs=self.controlnet_gs,
                )
            else:
                x = denoise(self.model, **inp_cond, timesteps=timesteps, guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs
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
