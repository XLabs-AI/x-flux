![FLUX Finetuning scripts](./assets/readme/dark/header-rev1.png)

<a href='https://replicate.com/lucataco/flux-controlnet'><img src='https://replicate.com/lucataco/flux-controlnet/badge'></a>

This repository provides training scripts for [Flux model](https://github.com/black-forest-labs/flux) by Black Forest Labs. <br/>
[XLabs AI](https://github.com/XLabs-AI) team is happy to publish fune-tuning Flux scripts, including:

- **LoRA** 🔥
- **ControlNet** 🔥
[<img src="https://github.com/XLabs-AI/x-flux/blob/main/assets/readme/light/join-our-discord-rev1.png?raw=true">](https://discord.gg/FHY2guThfy)

# ComfyUI

[See our github](https://github.com/XLabs-AI/x-flux-comfyui) for comfy ui workflows.
![Example Picture 1](https://github.com/XLabs-AI/x-flux-comfyui/blob/main/assets/image1.png?raw=true)

## Requirements
1. Python >= 3.10
2. PyTorch >= 2.1
3. HuggingFace CLI is required to download our models: ```huggingface-cli login```
# Installation Guide
1. Clone our repo:
```bash
git clone https://github.com/XLabs-AI/x-flux.git
```
2. Create new virtual environment:
```bash
python3 -m venv xflux_env
source xflux_env/bin/activate
```
3. Install our dependencies by running the following command:
```bash
pip install -r requirements.txt
```

# Training

We trained LoRA and ControlNet models using [DeepSpeed](https://github.com/microsoft/DeepSpeed)! <br/>
It's available for 1024x1024 resolution!

## Models

We trained **IP-Adapter**, **Canny ControlNet**, **Depth ControlNet**, **HED ControlNet** and **LoRA** checkpoints for [`FLUX.1 [dev]`](https://github.com/black-forest-labs/flux) <br/>
You can download them on HuggingFace:

- [flux-ip-adapter](https://huggingface.co/XLabs-AI/flux-ip-adapter)
- [flux-controlnet-collections](https://huggingface.co/XLabs-AI/flux-controlnet-collections)
- [flux-controlnet-canny](https://huggingface.co/XLabs-AI/flux-controlnet-canny)
- [flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora)
- [flux-lora-collections](https://huggingface.co/XLabs-AI/flux-lora-collection)
- [flux-furry-lora](https://huggingface.co/XLabs-AI/flux-furry-lora)

Also, our models are avaiable at [civit.ai](https://civitai.com/user/xlabs_ai)
### LoRA

```bash
accelerate launch train_flux_lora_deepspeed.py --config "train_configs/test_lora.yaml"
```

### ControlNet

```bash
accelerate launch train_flux_deepspeed_controlnet.py --config "train_configs/test_canny_controlnet.yaml"
```

## Training Dataset

Dataset has the following format for the training process:

```text
├── images/
│    ├── 1.png
│    ├── 1.json
│    ├── 2.png
│    ├── 2.json
│    ├── ...
```

### Example `images/*.json` file

A `.json` file contains "caption" field with a text prompt.

```json
{
    "caption": "A figure stands in a misty landscape, wearing a mask with antlers and dark, embellished attire, exuding mystery and otherworldlines"
}
```

## Inference

To test our checkpoints, you can use several options:
1. Launch adapters in ComfyUI with our workflows, [see our repo](https://github.com/XLabs-AI/x-flux-comfyui) for more details
2. Use main.py script with CLI commands
3. Use Gradio demo with simple UI

### Gradio
Launch gradio as follows:
```
python3 gradio_demo.py --ckpt_dir model_weights
```
Define `--ckpt_dir` as the folder location with the downloaded XLabs AI adapter weights (LoRAs, IP-adapter, ControlNets)
### IP-Adapter
```bash
python3 main.py \
 --prompt "wearing glasses" \
 --ip_repo_id XLabs-AI/flux-ip-adapter --ip_name flux-ip-adapter.safetensors --device cuda --use_ip \
 --width 1024 --height 1024 \
 --timestep_to_start_cfg 1 --num_steps 25 \
 --true_gs 3.5 --guidance 4 \
 --img_prompt assets/example_images/statue.jpg
```

### LoRA
![Example Picture 1](./assets/readme/examples/picture-5-rev1.png)
prompt: "A girl in a suit covered with bold tattoos and holding a vest pistol, beautiful woman, 25 years old, cool, future fantasy, turquoise & light orange ping curl hair"
![Example Picture 2](./assets/readme/examples/picture-6-rev1.png)
prompt: "A handsome man in a suit, 25 years old, cool, futuristic"

```bash
python3 main.py \
 --prompt "A cute corgi lives in a house made out of sushi, anime" \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name anime_lora.safetensors \
 --use_lora --width 1024 --height 1024
```
![Example Picture 3](./assets/readme/examples/result_14.png)


```bash
python3 main.py \
 --use_lora --lora_weight 0.7 \
 --width 1024 --height 768 \
 --lora_repo_id XLabs-AI/flux-lora-collection \
 --lora_name realism_lora.safetensors \
 --guidance 4 \
 --prompt "contrast play photography of a black female wearing white suit and albino asian geisha female wearing black suit, solid background, avant garde, high fashion"
```
![Example Picture 3](./assets/readme/examples/picture-7-rev1.png)

## Canny ControlNet V3
```bash
python3 main.py \
 --prompt "cyberpank dining room, full hd, cinematic" \
 --image input_canny1.png --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4
```
![Example Picture 1](./assets/readme/examples/canny_result1.png?raw=true)
```bash
python3 main.py \
 --prompt "handsome korean woman, full hd, cinematic" \
 --image input_canny2.png --control_type canny \
 --repo_id XLabs-AI/flux-controlnet-canny-v3 \
 --name flux-canny-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 4 --guidance 4
```
![Example Picture 1](./assets/readme/examples/canny_result2.png?raw=true)

## Depth ControlNet V3
```bash
python3 main.py \
 --prompt "handsome man in balenciaga style, fashion" \
 --image input_depth1.png --control_type depth \
 --repo_id XLabs-AI/flux-controlnet-depth-v3 \
 --name flux-depth-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024 --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 3
```
![Example Picture 2](./assets/readme/examples/depth_result1.png?raw=true)

```bash
python3 main.py \
 --prompt "a village in minecraft style, 3d, full hd" \
 --image input_depth2.png --control_type depth \
 --repo_id XLabs-AI/flux-controlnet-depth-v3 \
 --name flux-depth-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024 --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 3
```
![Example Picture 2](./assets/readme/examples/depth_result2.png?raw=true)

## HED ControlNet V3
```bash
 python3 main.py \
 --prompt "A beautiful woman with white hair and light freckles, her neck area bare and visible" \
 --image input_hed1.png --control_type hed \
 --repo_id XLabs-AI/flux-controlnet-hed-v3 \
 --name flux-hed-controlnet-v3.safetensors \
 --use_controlnet --model_type flux-dev \
 --width 1024 --height 1024  --timestep_to_start_cfg 1 \
 --num_steps 25 --true_gs 3.5 --guidance 4
```
![Example Picture 2](./assets/readme/examples/hed_result1.png?raw=true)

## Low memory mode

Use quantized version [Flux-dev-F8](https://huggingface.co/XLabs-AI/flux-dev-fp8) to achieve lower VRAM usage (22 GB) with `--offload` and `--model_type flux-dev-fp8` settings:
```bash
python3 main.py \
 --offload --model_type flux-dev-fp8 \
  --lora_repo_id XLabs-AI/flux-lora-collection --lora_name realism_lora.safetensors \
 --guidance 4 \
 --prompt "A handsome girl in a suit covered with bold tattoos and holding a pistol"
```
![Example Picture 0](./assets/readme/examples/picture-0-rev1.png)

## Accelerate Configuration Example

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 2
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false

```
## Models Licence

Our models fall under the [FLUX.1 [dev] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev) <br/> Our training and infer scripts under the Apache 2 License

## Near Updates

We are working on releasing new ControlNet weight models for Flux: **OpenPose**, **Depth** and more! <br/>
Stay tuned with [XLabs AI](https://github.com/XLabs-AI) to see **IP-Adapters** for Flux.

![Follow Our Updates](./assets/readme/dark/follow-cta-rev2.png)
