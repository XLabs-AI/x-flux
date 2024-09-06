import argparse
from PIL import Image
import os

from src.flux.xflux_pipeline import XFluxPipeline


def create_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt", type=str, required=True,
        help="The input text prompt"
    )
    parser.add_argument(
        "--neg_prompt", type=str, default="",
        help="The input text negative prompt"
    )
    parser.add_argument(
        "--img_prompt", type=str, default=None,
        help="Path to input image prompt"
    )
    parser.add_argument(
        "--neg_img_prompt", type=str, default=None,
        help="Path to input negative image prompt"
    )
    parser.add_argument(
        "--ip_scale", type=float, default=1.0,
        help="Strength of input image prompt"
    )
    parser.add_argument(
        "--neg_ip_scale", type=float, default=1.0,
        help="Strength of negative input image prompt"
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="A filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (IP-Adapter)"
    )
    parser.add_argument(
        "--ip_name", type=str, default=None,
        help="A IP-Adapter filename to download from HuggingFace"
    )
    parser.add_argument(
        "--ip_local_path", type=str, default=None,
        help="Local path to the model checkpoint (IP-Adapter)"
    )
    parser.add_argument(
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--lora_local_path", type=str, default=None,
        help="Local path to the model checkpoint (Controlnet)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_ip", action='store_true', help="Load IP model"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--num_images_per_prompt", type=int, default=1,
        help="The number of images to generate per prompt"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
    )
    parser.add_argument(
        "--width", type=int, default=1024, help="The width for generated image"
    )
    parser.add_argument(
        "--height", type=int, default=1024, help="The height for generated image"
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="The num_steps for diffusion process"
    )
    parser.add_argument(
        "--guidance", type=float, default=4, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3.5, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance"
    )
    parser.add_argument(
        "--save_path", type=str, default='results', help="Path to save"
    )
    return parser


def main(args):
    if args.image:
        image = Image.open(args.image)
    else:
        image = None

    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload)
    if args.use_ip:
        print('load ip-adapter:', args.ip_local_path, args.ip_repo_id, args.ip_name)
        xflux_pipeline.set_ip(args.ip_local_path, args.ip_repo_id, args.ip_name)
    if args.use_lora:
        print('load lora:', args.lora_local_path, args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(args.lora_local_path, args.lora_repo_id, args.lora_name, args.lora_weight)
    if args.use_controlnet:
        print('load controlnet:', args.local_path, args.repo_id, args.name)
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)

    image_prompt = Image.open(args.img_prompt) if args.img_prompt else None
    neg_image_prompt = Image.open(args.neg_img_prompt) if args.neg_img_prompt else None

    for _ in range(args.num_images_per_prompt):
        result = xflux_pipeline(
            prompt=args.prompt,
            controlnet_image=image,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            control_weight=args.control_weight,
            neg_prompt=args.neg_prompt,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            image_prompt=image_prompt,
            neg_image_prompt=neg_image_prompt,
            ip_scale=args.ip_scale,
            neg_ip_scale=args.neg_ip_scale,
        )
        if not os.path.exists(args.save_path):
            os.mkdir(args.save_path)
        ind = len(os.listdir(args.save_path))
        result.save(os.path.join(args.save_path, f"result_{ind}.png"))
        args.seed = args.seed + 1


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
