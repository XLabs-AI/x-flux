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
        "--lora_repo_id", type=str, default=None,
        help="A HuggingFace repo id to download model (LoRA)"
    )
    parser.add_argument(
        "--lora_name", type=str, default=None,
        help="A LoRA filename to download from HuggingFace"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device to use (e.g. cpu, cuda:0, cuda:1, etc.)"
    )
    parser.add_argument(
        "--offload", action='store_true', help="Offload model to CPU when not in use"
    )
    parser.add_argument(
        "--use_lora", action='store_true', help="Load Lora model"
    )
    parser.add_argument(
        "--use_controlnet", action='store_true', help="Load Controlnet model"
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to image"
    )
    parser.add_argument(
        "--lora_weight", type=float, default=0.9, help="Lora model strength (from 0 to 1.0)"
    )
    parser.add_argument(
        "--control_type", type=str, default="canny",
        choices=("canny", "openpose", "depth", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny"
    )
    parser.add_argument(
        "--model_type", type=str, default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)"
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
        "--guidance", type=float, default=3.5, help="The guidance for diffusion process"
    )
    parser.add_argument(
        "--seed", type=int, default=123456789, help="A seed for reproducible inference"
    )
    parser.add_argument(
        "--true_gs", type=float, default=3, help="true guidance"
    )
    parser.add_argument(
        "--timestep_to_start_cfg", type=int, default=100, help="timestep to start true guidance"
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

    xflux_pipeline = XFluxPipeline(args.model_type, args.device, args.offload, args.seed)
    if args.use_lora:
        print('load lora:', args.lora_repo_id, args.lora_name)
        xflux_pipeline.set_lora(None, args.lora_repo_id, args.lora_name, args.lora_weight)
    if args.use_controlnet:
        xflux_pipeline.set_controlnet(args.control_type, args.local_path, args.repo_id, args.name)

    result = xflux_pipeline(prompt=args.prompt,
                            controlnet_image=image,
                            width=args.width,
                            height=args.height,
                            guidance=args.guidance,
                            num_steps=args.num_steps,
                            true_gs=args.true_gs,
                            neg_prompt=args.neg_prompt,
                            timestep_to_start_cfg=args.timestep_to_start_cfg,
                            )
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    ind = len(os.listdir(args.save_path))
    result.save(os.path.join(args.save_path, f"result_{ind}.png"))


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
