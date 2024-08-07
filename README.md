![FLUX Finetuning scripts](./assets/readme/dark/header-rev1.png)

This repository provides training scripts for [Flux model](https://github.com/black-forest-labs/flux) by Black Forest Labs. <br/>
[XLabs AI](https://github.com/XLabs-AI) team is happy to publish fune-tuning Flux scripts, including:

- **LoRA** 🔥
- **ControlNet** 🔥

# Training

We trained LoRA and ControlNet models using [DeepSpeed](https://github.com/microsoft/DeepSpeed)! <br/>
Both of them are trained on 512x512 pictures, 1024x1024 is in progress.

## Models

We trained **Canny ControlNet** and **LoRA** checkpoints for [`FLUX.1 [dev]`](https://github.com/black-forest-labs/flux) <br/>
You can download them on HuggingFace:

- [flux-controlnet-canny](https://huggingface.co/XLabs-AI/flux-controlnet-canny)
- [flux-RealismLora](https://huggingface.co/XLabs-AI/flux-RealismLora)

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

To test our checkpoints, use commands presented below.

### LoRA

```bash
python3 demo_lora_inference.py \
    --repo_id XLabs-AI/flux-RealismLora \
    --prompt "A handsome girl in a suit covered with bold tattoos and holding a pistol. fantasy style, natural photo cinematic"
```
## Low memory mode

Use LoRA FP8 version based on [Flux-dev-F8](https://huggingface.co/XLabs-AI/flux-dev-fp8) with `--offload` setting to achieve lower VRAM usage (22 GB):
```bash
python3 demo_lora_inference.py \
    --repo_id XLabs-AI/flux-RealismLora \
    --prompt "A handsome girl in a suit covered with bold tattoos and holding a pistol. fantasy style, natural photo cinematic" --offload --name flux-dev-fp8
```
![Example Picture 0](./assets/readme/examples/picture-0-rev1.png)

### ControlNet (Canny)

```bash
python3 demo_controlnet_inference.py \
    --checkpoint controlnet.safetensors \
    --control_image "input_image.jpg" \
    --prompt "a bright blue bird in the garden, natural photo cinematic, MM full HD"
```

![Example Picture 1](./assets/readme/examples/picture-1-rev1.png)

```bash
python3 demo_controlnet_inference.py \
    --checkpoint controlnet.safetensors \
    --control_image "input_image.jpg" \
    --prompt "a dark evil mysterius house with ghosts, cinematic, MM full HD"
```

![Example Picture 2](./assets/readme/examples/picture-2-rev1.png)

```bash
python3 demo_controlnet_inference.py \
    --checkpoint controlnet.safetensors \
    --control_image "input_image.jpg" \
    --prompt "a handsome viking man with white hair, cinematic, MM full HD"
```

![Example Picture 3](./assets/readme/examples/picture-3-rev1.png)

```bash
python3 demo_controlnet_inference.py \
    --checkpoint controlnet.safetensors \
    --control_image "input_image.jpg" \
    --prompt "a oil painting woman sitting at chair and smiling, cinematic, MM full HD"
```

![Example Picture 4](./assets/readme/examples/picture-4-rev1.png)

## Requirements

Install our dependencies by running the following command:

```bash
pip3 install requirements.txt
```

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
