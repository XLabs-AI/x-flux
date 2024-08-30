from pathlib import Path

import torch
import gradio as gr

from src.flux.xflux_pipeline import XFluxPipeline


def create_demo(
        model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        offload: bool = False,
        ckpt_dir: str = "",
    ):
    xflux_pipeline = XFluxPipeline(model_type, device, offload)
    checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))

    with gr.Blocks() as demo:
        gr.Markdown(f"# Flux Adapters by XLabs AI - Model: {model_type}")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")

                with gr.Accordion("Generation Options", open=False):
                    with gr.Row():
                        width = gr.Slider(512, 2048, 1024, step=16, label="Width")
                        height = gr.Slider(512, 2048, 1024, step=16, label="Height")
                    neg_prompt = gr.Textbox(label="Negative Prompt", value="bad photo")
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="Number of steps")
                        timestep_to_start_cfg = gr.Slider(1, 50, 1, step=1, label="timestep_to_start_cfg")
                    with gr.Row():
                        guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="Guidance", interactive=True)
                        true_gs = gr.Slider(1.0, 5.0, 3.5, step=0.1, label="True Guidance", interactive=True)
                    seed = gr.Textbox(-1, label="Seed (-1 for random)")

                with gr.Accordion("ControlNet Options", open=False):
                    control_type = gr.Dropdown(["canny", "hed", "depth"], label="Control type")
                    control_weight = gr.Slider(0.0, 1.0, 0.8, step=0.1, label="Controlnet weight", interactive=True)
                    local_path = gr.Dropdown(checkpoints, label="Controlnet Checkpoint",
                        info="Local Path to Controlnet weights (if no, it will be downloaded from HF)"
                        )
                    controlnet_image = gr.Image(label="Input Controlnet Image", visible=True, interactive=True)

                with gr.Accordion("LoRA Options", open=False):
                    lora_weight = gr.Slider(0.0, 1.0, 0.9, step=0.1, label="LoRA weight", interactive=True)
                    lora_local_path = gr.Dropdown(
                        checkpoints, label="LoRA Checkpoint", info="Local Path to Lora weights"
                        )

                with gr.Accordion("IP Adapter Options", open=False):
                    image_prompt = gr.Image(label="image_prompt", visible=True, interactive=True)
                    ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="ip_scale")
                    neg_image_prompt = gr.Image(label="neg_image_prompt", visible=True, interactive=True)
                    neg_ip_scale = gr.Slider(0.0, 1.0, 1.0, step=0.1, label="neg_ip_scale")
                    ip_local_path = gr.Dropdown(
                        checkpoints, label="IP Adapter Checkpoint",
                        info="Local Path to IP Adapter weights (if no, it will be downloaded from HF)"
                        )
                generate_btn = gr.Button("Generate")

            with gr.Column():
                output_image = gr.Image(label="Generated Image")
                download_btn = gr.File(label="Download full-resolution")

        inputs = [prompt, image_prompt, controlnet_image, width, height, guidance,
                  num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                  neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                  lora_weight, local_path, lora_local_path, ip_local_path
                  ]
        generate_btn.click(
            fn=xflux_pipeline.gradio_generate,
            inputs=inputs,
            outputs=[output_image, download_btn],
        )


    return demo

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--ckpt_dir", type=str, default=".", help="Folder with checkpoints in safetensors format")
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload, args.ckpt_dir)
    demo.launch(share=args.share)
