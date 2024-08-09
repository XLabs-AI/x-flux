import os
import torch
import argparse
import numpy as np
from PIL import Image
from einops import rearrange

from image_datasets.canny_dataset import canny_processor, c_crop
from src.flux.sampling import denoise_controlnet, get_noise, get_schedule, prepare, unpack
from src.flux.util import (load_ae, load_clip, load_t5,
                           load_flow_model, load_controlnet, load_safetensors, load_from_repo_id)


def get_models(name: str, device: torch.device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    model = load_flow_model(name, device="cpu" if offload else device)
    ae = load_ae(name, device="cpu" if offload else device)
    controlnet = load_controlnet(name, device).to(torch.bfloat16)
    return model, ae, t5, clip, controlnet

def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--control_image", type=str, required=True,
        help="Path to the input image for control"
    )
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
        help="A HuggingFace repo id to download ControlNet model"
    )
    parser.add_argument(
        "--filename", type=str, default="controlnet.safetensors",
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
        "--output_dir", type=str, default="./controlnet_results/",
        help="The output directory where generation image is saved"
    )
    parser.add_argument(
        "--width", type=int, default=512, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=512, help="The height for generated image"
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

def preprocess_canny_image(image_path: str, width: int = 512, height: int = 512):
    image = Image.open(image_path)
    image = c_crop(image)
    image = image.resize((width, height))
    image = canny_processor(image)
    return image

def main(args):
    if args.checkpoint is None and args.repo_id is None:
        raise ValueError(
            "You must specify either args.checkpoint or args.repo_id to load ControlNet model"
        )
    name = "flux-dev"
    offload = args.offload
    is_schnell = name == "flux-schnell"
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    torch_device = torch.device(args.device)
    model, ae, t5, clip, controlnet = get_models(
        name,
        device=torch_device,
        offload=offload,
        is_schnell=is_schnell,
    )
    model = model.to(torch_device)
    if args.checkpoint is not None:
        if '.safetensors' in args.checkpoint:
            checkpoint = load_safetensors(args.checkpoint)
        else:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
    elif args.repo_id is not None:
        checkpoint = load_from_repo_id(args.repo_id, args.filename)

    controlnet.load_state_dict(checkpoint, strict=False)

    width = 16 * args.width // 16
    height = 16 * args.height // 16
    timesteps = get_schedule(
        args.num_steps,
        (width // 8) * (height // 8) // (16 * 16),
        shift=(not is_schnell),
    )
    filename = os.path.basename(args.control_image)
    canny_processed = preprocess_canny_image(args.control_image, width, height)
    canny_processed.save(os.path.join(args.output_dir, f"canny_processed_{filename}"))
    controlnet_cond = torch.from_numpy((np.array(canny_processed) / 127.5) - 1)
    controlnet_cond = controlnet_cond.permute(2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(torch_device)

    torch.manual_seed(args.seed)
    with torch.no_grad():
        x = get_noise(
            1, height, width, device=torch_device,
            dtype=torch.bfloat16, seed=args.seed
        )
        if offload:
            t5, clip = t5.to(torch_device), clip.to(torch_device)
        inp_cond = prepare(t5=t5, clip=clip, img=x, prompt=args.prompt)

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        x = denoise_controlnet(
            model, **inp_cond,
            controlnet=controlnet,
            timesteps=timesteps,
            guidance=args.guidance,
            controlnet_cond=controlnet_cond,
            #weight_type=torch.bfloat16,
        )
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
    output_path = os.path.join(args.output_dir, f"controlnet_result_{filename}")
    output_img.save(output_path)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
