import argparse
import sys
import os
import re
import time
from glob import iglob
from io import BytesIO

import torch
from dataclasses import dataclass

from einops import rearrange
from PIL import ExifTags, Image
from torchvision import transforms
from transformers import pipeline
from src.flux.modules.layers import DoubleStreamBlockLoraProcessor
from src.flux.sampling import denoise, get_noise, get_schedule, prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model, load_t5, load_safetensors, load_flow_model_quintized, load_from_repo_id)


def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    if 'fp8' in name:
        model = load_flow_model_quintized(name, device="cpu" if offload else device)
    else:
        model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    return model, ae, t5, clip

def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download LoRA model"
    )
    parser.add_argument(
        "--name", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model name to use (flux-dev, flux-dev-fp8, -flux-schnell)"
    )
    parser.add_argument(
        "--filename", type=str, default="lora.safetensors",
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./lora_results/",
        help="The output directory where generation image is saved"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=50, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    return parser


def main(args):
    if args.checkpoint is None and args.repo_id is None:
        raise ValueError(
            "You must specify either args.checkpoint or args.repo_id to load LoRA model"
        )
    name = args.name
    offload = args.offload
    is_schnell = name == "flux-schnell"
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    torch_device = torch.device(args.device)
    model, ae, t5, clip = get_models(
        name,
        device=torch_device,
        offload=False,
        is_schnell=is_schnell,
    )
    if args.checkpoint is not None:
        if '.safetensors' in args.checkpoint:
            checkpoint = load_safetensors(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
    elif args.repo_id is not None:
        checkpoint = load_from_repo_id(args.repo_id, args.filename)

    rank = get_lora_rank(checkpoint)
    lora_attn_procs = {}
    for name, _ in model.attn_processors.items():
        lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
        lora_state_dict = {}
        for k in checkpoint.keys():
            if name in k:
                lora_state_dict[k[len(name) + 1:]] = checkpoint[k]
        lora_attn_procs[name].load_state_dict(lora_state_dict)
        lora_attn_procs[name].to(torch_device)
    model.set_attn_processor(lora_attn_procs)


    width = 16 * args.width // 16
    height = 16 * args.height // 16

    torch.manual_seed(args.seed)
    with torch.no_grad():
        x = get_noise(
            1, height, width, device=torch_device,
            dtype=torch.bfloat16, seed=args.seed
        )
        timesteps = get_schedule(
            args.num_steps,
            x.shape[-1] * x.shape[-2] // (16 * 16),
            shift=(not is_schnell),
        )
        if offload:
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=args.prompt)

         # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        x = denoise(model, **inp_cond, timesteps=timesteps, guidance=args.guidance)

        if offload:
            model.cpu()
            torch.cuda.empty_cache()
            ae.decoder.to(x.device)

        x = unpack(x.float(), height, width)
        x = ae.decode(x)

        if args.offload:
            ae.decoder.cpu()
            torch.cuda.empty_cache()

    x1 = x.clamp(-1, 1)
    x1 = rearrange(x1[-1], "c h w -> h w c")
    output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
    output_idx = len(os.listdir(args.output_dir)) + 1
    output_path = os.path.join(args.output_dir, f"lora_result_{output_idx}.png")
    output_img.save(output_path)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
